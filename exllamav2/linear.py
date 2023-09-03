import torch
from exllamav2.module import ExLlamaV2Module
from torch import nn
import math
from exllamav2 import ext
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from safetensors import safe_open


def _tsize(st, key):

    tslice = st.get_slice(key)
    shape = tslice.get_shape()
    numel = 1
    for x in shape: numel *= x
    dtype = tslice.get_dtype()
    if dtype == "I32": return numel * 4
    elif dtype == "I16": return numel * 2
    elif dtype == "F16": return numel * 2
    else: raise ValueError("Unexpected datatype: " + key)


class ExLlamaV2Linear(ExLlamaV2Module):

    in_features: int
    out_features: int
    has_bias: bool

    linear: nn.Linear or None = None
    q_handle: int or None = None
    q_tensors: dict or None = None
    footprint: int

    name: str = "Linear"

    temp_dq: torch.tensor

    def __init__(self, model, key, in_features, out_features, has_bias):
        super().__init__(model, key)

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        self.temp_dq = None
        self.footprint = -1


    def load(self, w = None):

        if w is None: w = self.load_weight()
        if isinstance(w, dict):
            device_tensors = self.model.get_device_tensors(self.device_idx)
            device_tensors.begin_scratch_alloc()
            self.temp_dq = device_tensors.get_scratch_slice(self.temp_dq_size())
            self.q_handle = ext.make_q_matrix(w, self.temp_dq)
            self.q_tensors = w

        elif isinstance(w, nn.Parameter):
            self.linear = nn.Linear(self.in_features, self.out_features, self.has_bias, device = "meta", dtype = torch.float16)
            self.linear.weight = w



    def unload(self):

        del self.linear
        self.linear = None


    def get_weight(self):

        return self.linear.weight.data


    def weight_footprint(self):

        if self.footprint == -1:

            if self.key + ".weight" in self.model.config.tensor_file_map:
                filename = self.model.config.tensor_file_map[self.key + ".weight"]
                with safe_open(filename, framework="pt", device="cpu") as st:
                    self.footprint = 0
                    self.footprint += _tsize(st, self.key + ".weight")

            elif self.key + ".q_weight" in self.model.config.tensor_file_map:
                filename = self.model.config.tensor_file_map[self.key + ".q_weight"]
                with safe_open(filename, framework="pt", device="cpu") as st:
                    self.footprint = 0
                    self.footprint += _tsize(st, self.key + ".q_weight") + 128
                    self.footprint += _tsize(st, self.key + ".q_invperm") + 128
                    self.footprint += _tsize(st, self.key + ".q_scale") + 128
                    self.footprint += _tsize(st, self.key + ".q_scale_max") + 128
                    self.footprint += _tsize(st, self.key + ".q_groups") + 128
                    self.footprint += _tsize(st, self.key + ".q_invperm") + 128

            else:
                raise ValueError("Can't find tensors in model files.")

        return self.footprint


    def scratch_space(self):

        return self.temp_dq_size() + \
               self.temp_fwd_size()


    def temp_dq_size(self):

        return self.in_features * self.out_features * 2 + 128


    def temp_fwd_size(self):

        return self.out_features * self.model.config.max_input_len * self.model.config.max_batch_size * 4 + 128


    def forward(self, hidden_states, cache = None, attn_mask = None, past_len = None, intermediates = False, test = False):

        # test = True
        if self.q_handle is not None and not test:

            output_shape = hidden_states.shape[:-1] + (self.out_features,)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            # hidden_states = hidden_states[:, self.q_tensors["q_perm"]]
            output = torch.empty((hidden_states.shape[0], self.out_features), dtype = torch.half, device = self.device())
            ext_c.gemm_half_q_half(hidden_states, self.q_handle, output)

            hidden_states = output.view(output_shape)

        else:

            matrix = self.get_weight_tensor_dq()
            hidden_states = torch.matmul(hidden_states, matrix)

        # hidden_states = self.linear.forward(hidden_states)

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states


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