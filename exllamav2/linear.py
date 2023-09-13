import torch
import torch.nn.functional as F
from exllamav2.module import ExLlamaV2Module
from torch import nn
from exllamav2 import ext
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from safetensors import safe_open


class ExLlamaV2Linear(ExLlamaV2Module):

    in_features: int
    out_features: int
    has_bias: bool

    linear: nn.Linear or None = None
    q_handle: int or None = None
    q_tensors: dict or None = None

    name: str = "Linear"

    temp_dq: torch.tensor
    padding: int = 0

    def __init__(self, model, key, in_features, out_features, has_bias):
        super().__init__(model, key)

        self.padding = -out_features % 32

        self.in_features = in_features
        self.out_features = out_features + self.padding
        self.has_bias = has_bias
        self.temp_dq = None
        self.footprint = -1


    def load(self, w = None):

        if w is None: w = self.load_weight()
        if isinstance(w, dict):
            device_tensors = self.model.get_device_tensors(self.device_idx)
            device_tensors.begin_scratch_alloc()
            self.temp_dq = device_tensors.get_scratch_slice(self.temp_dq_size())
            self.q_tensors = w
            self.q_handle = ext.make_q_matrix(w, self.temp_dq)

        elif isinstance(w, nn.Parameter):
            if self.padding > 0: w = nn.Parameter(F.pad(w.data, (0, 0, 0, self.padding)).contiguous())
            self.linear = nn.Linear(self.in_features, self.out_features, self.has_bias, device = "meta", dtype = torch.float16)
            self.linear.weight = w


    def unload(self):

        del self.linear
        self.linear = None


    def get_weight(self):

        return self.linear.weight.data


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


    def forward(self, hidden_states, cache = None, attn_mask = None, past_len = None, intermediates = False, force_recons = False, force_cuda = False):

        # test = True
        if self.q_handle is not None and not force_recons:

            output_shape = hidden_states.shape[:-1] + (self.out_features,)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            # hidden_states = hidden_states[:, self.q_tensors["q_perm"]]
            output = torch.empty((hidden_states.shape[0], self.out_features), dtype = torch.half, device = self.device())
            ext_c.gemm_half_q_half(hidden_states, self.q_handle, output, force_cuda)

            hidden_states = output.view(output_shape)

        else:

            matrix = self.get_weight_tensor_dq()
            hidden_states = torch.matmul(hidden_states, matrix)

        # hidden_states = self.linear.forward(hidden_states)

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states


    def dump_group_info(self):

        if "q_groups" in self.q_tensors:

            groups = self.q_tensors["q_groups"].cpu()

            if "q_invperm" in self.q_tensors:
                height = self.q_tensors["q_invperm"].shape[0]
            else:
                height = self.q_tensors["q_weight"].shape[0] * 8

            groupsize = 1
            while groupsize * groups.shape[0] / 2 < height:
                groupsize *= 2;

            gis = f"gs: {groupsize}, "
            i = 0
            pg = 0
            gc = 0
            while i <= groups.shape[0]:
                g = groups[i].item() if i < groups.shape[0] else -1
                if g != pg:
                    if pg != 0:
                        gis += f"{pg}: {gc}, "
                        gc = 0
                    pg = g
                gc += 1
                i += 2

            return gis

        else:

            return "GPTQ"


    def get_weight_tensor_dq(self):

        if self.linear is not None:
            return self.linear.weight.data.T

        elif self.q_handle is not None:
            tensor = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.device())
            ext_c.reconstruct(self.q_handle, tensor)
            return tensor
            # ext_c.reconstruct(self.q_handle, self.temp_dq)
            # return self.temp_dq

        else:
            raise ValueError(f"Layer {self.key} has no data")


    def is_quant(self):

        return self.q_handle is not None