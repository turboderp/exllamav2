import torch
import torch.nn as nn
from exllamav2.config import ExLlamaV2Config
import json
import numpy as np


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


    def load_multi(self, keys, measure = False):

        tensors = {}
        submap = {}
        submap_i = {}
        size = 0

        for k in keys:
            ck = self.key + "." + k
            if ck in self.model.config.tensor_file_map:
                submap[k] = self.model.config.tensor_file_map[ck]

        for k, v in submap.items():
            if v not in submap_i:
                submap_i[v] = []
            submap_i[v].append(k)

        for v, ks in submap_i.items():
            with open(v, 'rb') as fp:
                header_size = np.fromfile(fp, dtype=np.int64, count=1).item()
                header_json = fp.read(header_size)
                byte_buffer_start = fp.tell()
                header = json.loads(header_json.decode('utf-8'))
                for k in ks:
                    meta = header[self.key + "." + k]
                    data_start, data_end = meta['data_offsets']
                    if measure:
                        size += data_end - data_start
                    else:
                        fp.seek(data_start+byte_buffer_start)
                        shape = meta['shape']
                        tensor_size = np.prod(shape)
                        dtype = {'I16': torch.int16, 'I32': torch.int32, 'F16': torch.float16, 'F32': torch.float32, 'F64': torch.float64, 'BF16': torch.bfloat16}[meta['dtype']]
                        buf = bytearray(fp.read(data_end-data_start))
                        t = torch.frombuffer(buf, dtype=dtype, count=tensor_size).reshape(shape)
                        tensors[k] = t.to(self.device())

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
