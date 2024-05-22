from __future__ import annotations
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import gc
import torch
import time


class Timer:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.interval = self.end_time - self.start_time


class SeqTensor:

    PAGE_SIZE = 256

    tensor: torch.Tensor
    seq_dim: int
    seq_len: int
    seq_cap: int

    def __init__(
        self,
        shape: tuple,
        dtype: torch.dtype,
        seq_dim: int,
        device: torch.device = "cpu",
        init_cap: int = -1
    ):
        if seq_dim < 0: seq_dim = len(shape) + seq_dim
        self.seq_dim = seq_dim
        self.seq_len = 0
        if init_cap == -1:
            init_cap = self.PAGE_SIZE
        else:
            init_cap = (init_cap // self.PAGE_SIZE + 1) * self.PAGE_SIZE
        shape = list(shape)
        shape[seq_dim] = self.seq_cap = init_cap
        shape = tuple(shape)
        self.tensor = torch.empty(shape, dtype = dtype, device = device)

    def __len__(self):
        return self.seq_len

    def __bool__(self):
        return self.seq_len > 0

    @staticmethod
    def from_tensor(tensor: torch.Tensor, seq_dim: int):
        s = SeqTensor(tensor.shape, tensor.dtype, seq_dim, tensor.device, init_cap = tensor.shape[seq_dim])
        s.append(tensor)
        return s

    def clone(self, drop: int | None = None):
        if drop and drop <= self.seq_len:
            return SeqTensor.from_tensor(self.torch_slice(None, self.seq_len - drop), self.seq_dim)
        else:
            return SeqTensor.from_tensor(self.torch(), self.seq_dim)

    def clear(self):
        self.seq_len = 0

    def set(self, new_data: SeqTensor | torch.tensor | None = None):
        self.clear()
        self.append(new_data)

    def append(self, new_data: SeqTensor | torch.tensor | None):
        if new_data is None: return
        if isinstance(new_data, SeqTensor):
            new_data = new_data.torch()
        new_len = new_data.shape[self.seq_dim]
        end_pos = self.seq_len + new_len
        if end_pos >= self.seq_cap:
            new_cap = (end_pos // self.PAGE_SIZE + 1) * self.PAGE_SIZE
            grow_shape = list(new_data.shape)
            grow_shape[self.seq_dim] = new_cap - self.seq_cap
            grow_shape = tuple(grow_shape)
            grow_tensor = torch.empty(grow_shape, dtype = self.tensor.dtype, device = self.tensor.device)
            self.tensor = torch.cat((self.tensor, grow_tensor), dim = self.seq_dim)
            self.seq_cap = new_cap
        s = self.tensor.narrow(self.seq_dim, self.seq_len, end_pos - self.seq_len)
        s.copy_(new_data)
        self.seq_len += new_len

    def truncate(self, new_len: int):
        assert new_len <= self.seq_len
        self.seq_len = new_len

    def torch(self):
        s = self.tensor.narrow(self.seq_dim, 0, self.seq_len)
        return s

    def slice(self, a: int | None, b: int | None):
        return SeqTensor.from_tensor(self.torch_slice(a, b), self.seq_dim)

    def torch_slice(self, a: int | None, b: int | None):
        if a is None and b is None:
            return self.torch()
        elif b is None:
            s = self.tensor.narrow(self.seq_dim, a, self.seq_len - a)
        elif a is None:
            s = self.tensor.narrow(self.seq_dim, 0, b)
        else:
            s = self.tensor.narrow(self.seq_dim, a, b - a)
        return s


def get_basic_progress():
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width = None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    return progress


def list_live_tensors():

    tensors = {}
    gc.collect()
    torch.cuda.empty_cache()

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                d = str(obj.size()) + ", " + str(obj.dtype) + ", " + str(obj.device)
                if d in tensors.keys():
                    tensors[d] += 1
                else:
                    tensors[d] = 1
        except:
            pass

    print("-----------")
    for k, v in tensors.items():
        print(f"{v} : {k}")


snapshot = {}

def set_snapshot():
    global snapshot

    snapshot = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                d = str(obj.size()) + ", " + str(obj.dtype) + ", " + str(obj.device)
                if d in snapshot.keys():
                    snapshot[d] += 1
                else:
                    snapshot[d] = 1
        except:
            pass


def diff_snapshot():
    global snapshot

    new_tensors = {}
    snapshot_copy = snapshot.copy()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                d = str(obj.size()) + ", " + str(obj.dtype) + ", " + str(obj.device)
                if d in snapshot_copy:
                    if snapshot_copy[d] == 1:
                        del snapshot_copy[d]
                    else:
                        snapshot_copy[d] -= 1
                else:
                    if d in new_tensors:
                        new_tensors[d] += 1
                    else:
                        new_tensors[d] = 1
        except:
            pass

    print("-----------")
    print("-- New tensors")
    for k, v in new_tensors.items(): print(f"{v} : {k}")
    print("-----------")
    print("-- Removed tensors")
    for k, v in snapshot_copy.items(): print(f"{v} : {k}")


def print_vram_usage():

    torch.cuda.reset_peak_memory_stats("cuda:0")
    mem_this = torch.cuda.max_memory_allocated("cuda:0")
    print(f"Peak memory: {mem_this / (1024 ** 2):,.2f} MB")


def print_vram_usage_peak():

    mem_this = torch.cuda.max_memory_allocated("cuda:0")
    print(f"Peak memory: {mem_this / (1024 ** 2):,.2f} MB")


