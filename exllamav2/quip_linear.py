import torch
from exllamav2.module import ExLlamaV2Module
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
import torch
from quip.matmul_had import get_hadK
from quip import codebook

class QuipLinear(ExLlamaV2Module):

    in_features: int
    out_features: int

    outlier_channel_split: bool
    rescale_WH: float or None
    scaleWH: torch.Tensor or None
    idx_dtype: torch.dtype
    codesz: int
    had_left: torch.FloatTensor
    K_left: int
    had_right: torch.FloatTensor
    K_right: int
    Qidxs: torch.Tensor or None
    SU: torch.Tensor or None
    SV: torch.Tensor or None
    Wscale: torch.Tensor or None
    packsz: int

    name: str = "QuipLinear"

    lora_a_tensors: dict
    lora_b_tensors: dict

    def __init__(self, model, key, in_features, out_features):
        super().__init__(model, key)

        self.in_features = in_features
        self.out_features = out_features
        self.outlier_channel_split = model.config.quip_params['outlier_channel_split']
        self.rescale_WH = model.config.quip_params['rescale_WH']
        self.idx_dtype = {
            'torch.int32': torch.int32,
            'torch.int16': torch.int16,
            'torch.uint8': torch.uint8,
            }[model.config.quip_params['idx_dtype']]
        self.codesz = model.config.quip_params['codesz']
        self.packsz = model.config.quip_params['packsz'] if 'packsz' in model.config.quip_params['packsz'] else 1
        self.packed = (self.packsz != 1)

        if self.outlier_channel_split: self.ocs_dupe_inds = torch.arange(in_features)

        self.scaleWH = torch.ones(in_features) if self.rescale_WH else None

        self.had_left, self.K_left = get_hadK(in_features)
        self.had_right, self.K_right = get_hadK(out_features)


    def load(self, w = None):
        if w is None: w = self.load_weight()
        self.Qidxs = torch.tensor(w['Qidxs'], dtype=self.idx_dtype, device=self.device_idx).reshape(self.out_features, self.in_features // (self.codesz*self.packsz))
        self.SU = torch.tensor(w['SU'], device=self.device_idx)
        self.SV = torch.tensor(w['SV'], device=self.device_idx)
        self.Wscale = torch.tensor(w['Wscale'], device=self.device_idx)
        self.cookbook = codebook.get_quantized_class(w['codebook_id'])(self.device_idx)


    def unload(self):
        del self.Qidxs
        del self.codebook_id
        del self.SU
        del self.SV
        del self.Wscale
        self.Qidxs = None
        self.codebook_id = None
        self.SU = None
        self.SV = None
        self.Wscale = None


    def get_weight(self):
      raise ValueError(f"QuiP Laye does not support get_weight.")


    def scratch_space_fixed(self):

        return self.temp_dq_size() + \
               self.temp_fwd_size()


    def scratch_space(self):

        return self.temp_dq_size() + \
               self.temp_fwd_size()


    def temp_dq_size(self):

        return self.in_features * self.out_features * 2 + 128


    def temp_fwd_size(self):

        return self.out_features * self.model.config.max_input_len * self.model.config.max_batch_size * 4 + 128


    def forward(self, hidden_states, cache = None, attn_mask = None, past_len = None, intermediates = False, loras = None, force_recons = False, force_cuda = False):
        lora_a = None
        lora_b = None
        if loras is not None:
            for lora in loras:
                lora_a = self.lora_a_tensors[lora] if lora in self.lora_a_tensors else None
                lora_b = self.lora_b_tensors[lora] if lora in self.lora_b_tensors else None

        if self.outlier_channel_split: hidden_states = hidden_states[..., self.ocs_dupe_inds]

        return self.cookbook(
            hidden_states,
            self.Qidxs, self.SU, self.SV, self.Wscale,
            self.had_left, self.had_right, self.K_left, self.K_right,
            A=lora_a, B=lora_b,
            rescale_WH=self.rescale_WH, scaleWH=self.scaleWH, packed=self.packed)

    def get_weight_tensor_dq(self):
      raise ValueError(f"QuiP Layer {self.key} does not support.")


    def is_quant(self):
        return True