import gc
import torch

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


