
import torch


# On some setups Torch will attempt to use GPU peer-to-peer copies even when they are not supported. This is either
# a driver issue, a bug in Torch, or both. Either way, the result is that .to() will create an empty tensor on the
# target device and silently fail to copy any data into it. This is a workaround.

disable_peer_copy = False
tested_peer_copy = False

def test_gpu_peer_copy():
    global disable_peer_copy, tested_peer_copy

    if tested_peer_copy: return
    tested_peer_copy = True

    num_dev = torch.cuda.device_count()
    for i in range(num_dev):
        dev_i = f"cuda:{i}"
        for j in range(i + 1, num_dev):
            dev_j = f"cuda:{j}"

            a = torch.randn(5, device = dev_i) + 123.0
            b = a.to(dev_j)
            c = b.to(dev_i)

            if not torch.all(a == c):
                disable_peer_copy = True


def safe_move_tensor(tensor, device):
    global disable_peer_copy

    # Accept torch.device or string

    device = torch.device(device)

    # No move

    if tensor.device == device:
        return tensor

    # Test tensor.to (once) and if it seems to be working, let Torch decide

    test_gpu_peer_copy()

    if not disable_peer_copy:
        return tensor.to(device)

    # Move tensor via CPU if source and dest are distinct CUDA devices

    if tensor.device.type == device.type == 'cuda' and tensor.device != device:
        return tensor.cpu().to(device)

    # Move to/from CPU

    return tensor.to(device)