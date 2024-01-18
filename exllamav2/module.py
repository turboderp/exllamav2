import torch
import torch.nn as nn
from exllamav2.config import ExLlamaV2Config
from exllamav2.fasttensors import STFile
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


    def numel(self):

        return 0


    def device(self):

        return _torch_device(self.device_idx)


    def load_multi(self, keys, override_key = None, measure = False):

        tensors = {}
        submap = {}
        submap_i = {}
        size = 0

        key = self.key if override_key is None else override_key

        for k in keys:
            ck = key + "." + k
            if ck in self.model.config.tensor_file_map:
                submap[k] = self.model.config.tensor_file_map[ck]

        for k, v in submap.items():
            if v not in submap_i:
                submap_i[v] = []
            submap_i[v].append(k)

        for v, ks in submap_i.items():
            stfile = STFile.open(v, fast = self.model.config.fasttensors)
            for k in ks:
                if measure:
                    size += stfile.measure(key + "." + k)
                else:
                    tensors[k] = stfile.get_tensor(key + "." + k, device = self.device())

            # with safe_open(v, framework="pt", device="cpu") as st:
            #     for k in ks:
            #         if measure:
            #             size += _tsize(st, key + "." + k)
            #         else:
            #             tensors[k] = st.get_tensor(key + "." + k).to(self.device())

        return size if measure else tensors


    def load_weight(self, override_key = None):

        key = self.key if override_key is None else override_key

        # EXL2

        if key + ".q_weight" in self.model.config.tensor_file_map:
            qtensors = self.load_multi(["q_weight", "q_invperm", "q_scale", "q_scale_max", "q_groups", "q_perm"], override_key = override_key)
            qtensors["q_perm"] = torch.argsort(qtensors["q_invperm"]).to(torch.int)
            return qtensors

        # GPTQ

        if key + ".qweight" in self.model.config.tensor_file_map:
            qtensors = self.load_multi(["qweight", "qzeros", "scales", "g_idx"], override_key = override_key)
            qtensors["scales"] = qtensors["scales"].half()
            return qtensors

        # Torch

        if key + ".weight" in self.model.config.tensor_file_map:
            tensor = self.load_multi(["weight"], override_key = override_key)["weight"]
            tensor = tensor.half()
            return nn.Parameter(tensor)

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

            # Error

            else: raise ValueError("Unknown tensor type: " + self.key)

        return self.footprint


    def set_device_idx(self, idx):

        self.device_idx = idx


    def is_quant(self):
        return False


    def reload(self):
        self.unload()
        self.load()
