from __future__ import annotations
import torch
from safetensors import safe_open
import numpy as np
import json
from exllamav2.ext import exllamav2_ext as ext_c
import os

def convert_dtype(dt: str):
    """
    :param dt:
        Datatype string as used by safetensors

    :return:
        Torch type, element size in bytes
    """
    if dt == "I32": return torch.int, 4
    elif dt == "I16": return torch.short, 2
    elif dt == "F16": return torch.float16, 2
    elif dt == "BF16": return torch.bfloat16, 2
    elif dt == "F32": return torch.float, 4
    else: raise ValueError(f"Unknown dtype {dt}")

global_stfiles = []
global_cm = {}

def cleanup_stfiles():
    global global_stfiles, global_cm

    for stf in global_stfiles:
        stf.close()
    global_stfiles = []

    for f in global_cm.values():
        f.__exit__(None, None, None)
    global_cm = {}

    ext_c.safetensors_free_pinned_buffer()


class STFile:

    filename: str
    header: dict
    header_size: int
    metadata: object
    handle: int
    fast: bool
    st_context = None

    def __init__(self,
                 filename: str,
                 fast: bool = True):
        global global_stfiles

        self.metadata = None
        self.handle = 0
        self.filename = filename

        self.read_dict()

        if fast and os.name == "nt":
            print(" !! Warning, fasttensors disabled on Windows")
            fast = False

        self.fast = fast
        if self.fast:
            self.handle = ext_c.safetensors_open(filename)

        global_stfiles.append(self)


    @staticmethod
    def open(filename,
             fast = True) -> STFile:
        """
        Open safetensors file, scan header and retain handle.

        :param filename:
            Filename

        :param fast:
            Use fast (direct I/O) codepath

        :return:
            STFile object
        """

        global global_stfiles
        for f in global_stfiles:
            if f.filename == filename: return f
        return STFile(filename, fast)


    def close(self):
        """
        Close file handle (if necessary)
        """
        if self.fast: ext_c.safetensors_close(self.handle)


    def read_dict(self):
        with open(self.filename, "rb") as fp:
            header_size = np.fromfile(fp, dtype = np.int64, count = 1).item()
            header_json = fp.read(header_size)
            self.header = json.loads(header_json.decode("utf-8"))
            self.header_size = fp.tell()
            if "__metadata__" in self.header:
                self.metadata = self.header["__metadata__"]
                del self.header["__metadata__"]


    def get_dict(self) -> dict:
        return self.header


    def get_metadata(self) -> object:
        return self.metadata


    def measure(self, key):
        """
        :param key:
            Tensor key

        :return:
            Byte size of tensor
        """
        v = self.header[key]
        data_offsets = v["data_offsets"]
        length = data_offsets[1] - data_offsets[0]
        return length


    def get_cm(self, device):
        global global_cm

        cm_key = self.filename + "::" + device

        if cm_key in global_cm:
            return global_cm[cm_key]

        f = safe_open(self.filename, framework = "pt", device = device)
        f.__enter__()
        global_cm[cm_key] = f
        return f


    def get_tensor(self,
                   key: str,
                   device,
                   not_fast: bool = False) -> torch.Tensor:

        if not_fast or not self.fast:
            f = self.get_cm(device)
            # with safe_open(self.filename, framework = "pt", device = device) as f:
            return f.get_tensor(key)

        v = self.header[key]
        dtt, dts = convert_dtype(v["dtype"])
        sh = v["shape"]
        data_offsets = v["data_offsets"]
        offset = data_offsets[0] + self.header_size
        length = data_offsets[1] - data_offsets[0]
        assert np.prod(sh) * dts == length, f"Tensor shape doesn't match storage size: {key}"

        tensor = torch.empty(sh, device = device, dtype = dtt)
        ext_c.safetensors_load(self.handle, tensor, offset, length)
        return tensor
