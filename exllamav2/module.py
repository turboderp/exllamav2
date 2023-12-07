import torch
import torch.nn as nn
from exllamav2.config import ExLlamaV2Config
from safetensors import safe_open


def _torch_device(idx):
    if idx == -1: return "cpu"
    return f"cuda:{idx}"


def _tsize(st, key):

    tslice = st.get_slice(key)
    shape = tslice.get_shape()
    numel = 1
    for x in shape: numel *= x
    dtype = tslice.get_dtype()
    if dtype == "I32": return numel * 4
    elif dtype == "I16": return numel * 2
    elif dtype == "F16": return numel * 2
    elif dtype == "BF16": return numel * 2
    elif dtype == "F32": return numel * 4
    else: raise ValueError(f"Unexpected datatype {dtype}: {key}")


class ExLlamaV2Module:

    model = None
    config: ExLlamaV2Config
    key: str
    device_idx: int
    footprint: int

    def __init__(self, model, key):

        self.model = model
        self.key = key
        self.footprint = -1


    def device(self):

        return _torch_device(self.device_idx)


    def load_multi(self, keys, measure = False, key_postfix = None):

        tensors = {}
        submap = {}
        submap_i = {}
        size = 0
        
        key = self.key if key_postfix is None else self.key + "." + key_postfix
        for k in keys:
            ck = key + "." + k
            if ck in self.model.config.tensor_file_map:
                submap[k] = self.model.config.tensor_file_map[ck]

        for k, v in submap.items():
            if v not in submap_i:
                submap_i[v] = []
            submap_i[v].append(k)

        for v, ks in submap_i.items():
            with safe_open(v, framework="pt", device="cpu") as st:
                for k in ks:
                    if measure:
                        size += _tsize(st, key + "." + k)
                    else:
                        # tensors[k] = st.get_tensor(key + "." + k)
                        tensors[k] = st.get_tensor(key + "." + k).to(self.device())

        return size if measure else tensors


    def load_weight(self):

        # EXL2

        if self.key + ".q_weight" in self.model.config.tensor_file_map:
            qtensors = self.load_multi(["q_weight", "q_invperm", "q_scale", "q_scale_max", "q_groups", "q_perm"])
            qtensors["q_perm"] = torch.argsort(qtensors["q_invperm"]).to(torch.int)
            return qtensors

        # GPTQ

        if self.key + ".qweight" in self.model.config.tensor_file_map:
            qtensors = self.load_multi(["qweight", "qzeros", "scales", "g_idx"])
            return qtensors

        # Torch

        if self.key + ".weight" in self.model.config.tensor_file_map:
            tensor = self.load_multi(["weight"])["weight"]
            tensor = tensor.half()
            return nn.Parameter(tensor)

        # QuiP
        
        if self.model.config.is_quip:
            if self.key + ".Qidxs" in self.model.config.tensor_file_map:
                return self.load_multi(["Qidxs", "SU", "SV", "Wscale", "codebook_id"])
            elif self.__class__.__name__ == "ExLlamaV2MLP" and self.key + '.mlp.down_scale' in self.model.config.tensor_file_map:
                return self.load_multi(["down_scale", "up_scale", "gate_scale"], key_postfix="mlp")
            elif self.__class__.__name__ == "ExLlamaV2Attention" and self.key + '.self_attn.k_scale' in self.model.config.tensor_file_map:
                return self.load_multi(["k_scale", "q_scale", "o_scale", "v_scale"], key_postfix="self_attn")
            
        # No weights found for key

        return None


    def weight_footprint(self):

        if self.footprint == -1:

            # EXL2

            if self.key + ".q_weight" in self.model.config.tensor_file_map:
                self.footprint = self.load_multi(["q_weight", "q_invperm", "q_scale", "q_scale_max", "q_groups", "q_perm", "q_perm"], measure = True)

            # GPTQ

            elif self.key + ".qweight" in self.model.config.tensor_file_map:
                self.footprint = self.load_multi(["qweight", "qzeros", "scales", "g_idx"], measure = True)

            # Torch

            elif self.key + ".weight" in self.model.config.tensor_file_map:
                self.footprint = self.load_multi(["weight"], measure = True)

            # QuiP
            
            elif self.key + ".Qidxs" in self.model.config.tensor_file_map:
                self.footprint = self.load_multi(["Qidxs", "SU", "SV", "Wscale", "codebook_id", "down_scale", "up_scale", "gate_scale", "k_scale", "q_scale", "o_scale", "v_scale"], measure = True)

            # Error

            else: raise ValueError("Unknown tensor type: " + self.key)

        return self.footprint


    def set_device_idx(self, idx):

        self.device_idx = idx


    def is_quant(self):
        return False
