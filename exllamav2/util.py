from __future__ import annotations
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import gc, subprocess, time, os, json
import torch


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


def cuda_sync_active():
    """
    Calling torch.cuda.synchronize() will create a CUDA context on CUDA:0 even if that device is not being used.
    This function synchronizes only devices actively used by Torch in the current process.
    """
    for device_id in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{device_id}')
        if torch.cuda.memory_allocated(device) > 0:
            torch.cuda.synchronize(device)


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


def get_visible_devices():

    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if visible_devices is None:
        return None
    return [int(dev) for dev in visible_devices.split(',')]


def get_nvidia_gpu_memory(visible_devices=None):

    """
    Get the current GPU usage for NVIDIA GPUs.
    TODO: Find a better way to respect CUDA_VISIBLE_DEVICES, perhaps pynvml?
    """

    result = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=index,memory.total,memory.used,memory.free',
            '--format=csv,nounits,noheader'
        ],
        stdout = subprocess.PIPE, encoding = 'utf-8'
    )

    gpu_memory = {}
    for line in result.stdout.strip().split('\n'):
        index, total, used, free = map(int, line.split(','))
        if visible_devices is None or index in visible_devices:
            gpu_memory[index] = {'total': total, 'used': used, 'free': free}
    if visible_devices is not None:
        gpu_memory = { idx: gpu_memory[d] for idx, d in enumerate(visible_devices) }

    return gpu_memory


def get_amd_gpu_memory():

    """
    Get the current GPU usage for AMD GPUs.
    TODO: Test on ROCm
    """

    result = subprocess.run(
        [
            'rocm-smi',
            '--showmeminfo',
            'vram',
            '--json'
        ],
        stdout=subprocess.PIPE, encoding='utf-8'
    )

    data = json.loads(result.stdout)

    gpu_memory = {}
    for gpu in data['card']:
        index = gpu['card_id']
        total = int(gpu['vram_total'])
        used = int(gpu['vram_used'])
        free = int(gpu['vram_free'])
        gpu_memory[index] = {'total': total, 'used': used, 'free': free}

    return gpu_memory


def get_all_gpu_memory():

    """
    Get the current GPU usage for both NVIDIA and AMD GPUs.
    """

    gpu_memory = {}
    visible_devices = get_visible_devices()

    try:
        nvidia_memory = get_nvidia_gpu_memory(visible_devices)
        gpu_memory.update(nvidia_memory)
    except:
        pass
        # print("nvidia-smi not found. Skipping NVIDIA GPU check.")

    try:
        amd_memory = get_amd_gpu_memory()
        gpu_memory.update(amd_memory)
    except:
        pass
        # print("rocm-smi not found. Skipping AMD GPU check.")  # TODO: test on AMD

    assert gpu_memory, \
        "Unable to read available VRAM from either nvidia-smi or rocm-smi"

    return gpu_memory


def integer_split(x, split: list[int], minimum: int = 0) -> list[int]:

    """
    Precisely split x integer into portions according to given ratio, ensuring sum(portions) == x
    """

    sum_split = sum(split)
    portions = [int(x * p / sum_split) for p in split]
    remaining = x - sum(portions)
    remainders = [(x * p / sum_split) - initial for p, initial in zip(split, portions)]
    for i in range(remaining):
        max_index = remainders.index(max(remainders))
        portions[max_index] += 1
        remainders[max_index] -= 1
    adjust = sum((p if p < minimum else 0) for p in portions)
    portions = [(p if p >= minimum else 0) for p in portions]
    for i in range(adjust):
        min_index = min((i for i, v in enumerate(portions) if v != 0), key = lambda i: portions[i], default = -1)
        portions[min_index] += 1
    return portions


def unpack_4bit(packed: torch.Tensor):
    """
    :param packed:
        (m, n // 8) tensor, dtype torch.int32/uint32, packed 4-bit ints
    :return:
        (m, n) tensor, dtype torch.int8
    TODO: CUDA kernel for this
    """

    m, n8 = packed.shape
    n = n8 * 8
    assert packed.dtype in [torch.int32]

    # packed = packed.view(torch.uint32)
    unpacked = torch.empty((m, n), dtype = torch.uint8, device = packed.device)
    for i in range(8):
        unpacked[:, i::8] = (packed >> (i * 4)) & 0xF
    return unpacked


def pack_4bit(unpacked: torch.Tensor):
    """
    :param unpacked:
        (m, n) tensor, dtype torch.int8, packed 4-bit ints
    :return:
        (m, n, // 8) tensor, dtype torch.int32
    TODO: CUDA kernel for this
    """

    m, n = unpacked.shape
    assert n % 8 == 0
    assert unpacked.dtype == torch.uint8

    packed = torch.zeros((m, n // 8), dtype = torch.int64, device = unpacked.device)
    for i in range(8):
        packed |= (unpacked[:, i::8].to(torch.int64) << (i * 4))
    packed = packed.to(torch.int32)
    return packed