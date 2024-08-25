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
global_tensorcache = []
num_cached_tensors = 4

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
    handle_fb: int
    fast: bool
    st_context = None
    tensor_remap: dict | None

    def __init__(self,
                 filename: str,
                 fast: bool = True,
                 keymap: list[tuple[str, str]] = None):

        global global_stfiles

        self.metadata = None
        self.handle = 0
        self.handle_fb = 0
        self.filename = filename
        self.st_context = None
        self.tensor_remap = None

        self.read_dict()

        if keymap:
            self.tensor_remap = {}
            nheader = {}
            for key in self.header.keys():
                nkey = key
                for z in keymap:
                    if z[0].startswith("$") and nkey.startswith(z[0][1:]):
                        nkey = ("$" + nkey).replace(z[0], z[1])
                    else:
                        nkey = nkey.replace(z[0], z[1])
                nheader[nkey] = self.header[key]
                self.tensor_remap[nkey] = key
            self.header = nheader

        # TODO: Create platform-independent multithreaded loader to replace direct IO loader on Linux and fallback
        #   cstdio loader on Windows

        if fast and os.name == "nt":
            self.fast_fb = True
            fast = False
        else:
            self.fast_fb = False

        self.fast = fast
        if self.fast:
            self.handle = ext_c.safetensors_open(filename)
        if self.fast_fb:
            self.handle_fb = ext_c.safetensors_open_fb(filename)

        global_stfiles.append(self)


    @staticmethod
    def open(filename,
             fast = True,
             keymap: list[tuple[str, str]] = None) -> STFile:
        """
        Open safetensors file, scan header and retain handle.

        :param filename:
            Filename

        :param fast:
            Use fast (direct I/O) codepath

        :param keymap:
            List of (a, b) tuples for string replacements in key index

        :return:
            STFile object
        """

        global global_stfiles
        for f in global_stfiles:
            if f.filename == filename: return f
        return STFile(filename, fast, keymap)


    def close(self):
        """
        Close file handle (if necessary)
        """
        if self.fast: ext_c.safetensors_close(self.handle)
        if self.fast_fb: ext_c.safetensors_close_fb(self.handle_fb)


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
                   not_fast: bool = False,
                   cached: bool = False,
                   out_dtype = None) -> torch.Tensor:
        global global_tensorcache

        torch.cuda.synchronize()

        if self.tensor_remap and (not_fast or not self.fast):
            key = self.tensor_remap[key]

        if cached:
            cachekey = self.filename + "::" + key + "::" + device
            for (k, v) in global_tensorcache:
                if k == cachekey: return v

        if not_fast or (not self.fast and not self.fast_fb):

            # with safe_open(self.filename, framework = "pt", device = device) as f:
            f = self.get_cm(device)
            tensor = f.get_tensor(key)

        elif self.fast_fb:

            h = self.header[key]
            dtype, esize = convert_dtype(h["dtype"])
            beg, end = h["data_offsets"]
            size = end - beg
            numel = size // esize
            shape = h["shape"]
            if device != "cpu":
                torch.cuda.set_stream(torch.cuda.default_stream(device))
            tensor = torch.zeros(shape, dtype = dtype, device = device)
            assert tensor.is_contiguous, "Non-contiguous tensor"
            ext_c.safetensors_read_fb(self.handle_fb, beg + self.header_size, size, tensor)

        else:

            v = self.header[key]
            dtt, dts = convert_dtype(v["dtype"])
            sh = v["shape"]
            data_offsets = v["data_offsets"]
            offset = data_offsets[0] + self.header_size
            length = data_offsets[1] - data_offsets[0]
            assert np.prod(sh) * dts == length, f"Tensor shape doesn't match storage size: {key}"
            if device != "cpu":
                torch.cuda.set_stream(torch.cuda.default_stream(device))
            tensor = torch.empty(sh, device = device, dtype = dtt)
            ext_c.safetensors_load(self.handle, tensor, offset, length)

        if out_dtype:
            tensor = tensor.to(out_dtype)

        if cached:
            if len(global_tensorcache) >= num_cached_tensors:
                global_tensorcache = global_tensorcache[1:]
            global_tensorcache.append((cachekey, tensor))

        torch.cuda.synchronize()

        return tensor
