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
    alt_key: str = None
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


    def load_multi(self, key, keys, override_key = None, measure = False):

        tensors = {}
        submap = {}
        submap_i = {}
        size = 0

        # key = self.key if override_key is None else override_key

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

        if override_key is not None:
            keys = [override_key]
        else:
            keys = [self.key]
            if self.alt_key is not None:
                keys += [self.alt_key]

        for key in keys:

            # EXL2

            if key + ".q_weight" in self.model.config.tensor_file_map:
                qtensors = self.load_multi(key, ["q_weight", "q_invperm", "q_scale", "q_scale_max", "q_groups", "q_perm", "bias"], override_key = override_key)
                qtensors["q_perm"] = torch.argsort(qtensors["q_invperm"]).to(torch.int)
                return qtensors

            # GPTQ

            if key + ".qweight" in self.model.config.tensor_file_map:
                qtensors = self.load_multi(key, ["qweight", "qzeros", "scales", "g_idx", "bias"], override_key = override_key)
                if "bias" in qtensors and torch.all(qtensors["bias"].eq(0)):
                    del qtensors["bias"]
                qtensors["scales"] = qtensors["scales"].half()
                return qtensors

            # Torch

            if key + ".weight" in self.model.config.tensor_file_map:
                if key + ".bias" in self.model.config.tensor_file_map:
                    tensors = self.load_multi(key, ["weight", "bias"], override_key = override_key)
                    tensor = tensors["weight"].half()
                    bias = tensors["bias"].half()
                    return nn.Parameter(tensor), nn.Parameter(bias)
                else:
                    tensors = self.load_multi(key, ["weight"], override_key = override_key)
                    tensor = tensors["weight"].half()
                    return nn.Parameter(tensor)

            # No weights found for key

        return None


    def weight_footprint(self):

        if self.footprint == -1:

            keys = [self.key]
            if self.alt_key is not None:
                keys += [self.alt_key]

            for key in keys:

                # EXL2

                if key + ".q_weight" in self.model.config.tensor_file_map:
                    self.footprint = self.load_multi(key, ["q_weight", "q_invperm", "q_scale", "q_scale_max", "q_groups", "q_perm", "q_perm"], measure = True)

                # GPTQ

                elif key + ".qweight" in self.model.config.tensor_file_map:
                    self.footprint = self.load_multi(key, ["qweight", "qzeros", "scales", "g_idx"], measure = True)

                # Torch

                elif key + ".weight" in self.model.config.tensor_file_map:
                    self.footprint = self.load_multi(key, ["weight"], measure = True)

                if self.footprint != -1: break

            # Error

            if self.footprint == -1:
                raise ValueError("Unknown tensor type: " + self.key)

        return self.footprint


    def set_device_idx(self, idx):

        self.device_idx = idx


    def is_quant(self):
        return False


    def reload(self):
        self.unload()
        self.load()
