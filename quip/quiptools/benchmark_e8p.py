import torch
import quiptools_cuda


def benchmark():
    torch.manual_seed(42)
    M = 1
    N = 12288
    K = 4096

    x = torch.randn((M, K), dtype=torch.float32, device="cuda")
    Qidxs = torch.randint(1<<15, (N, K//8), dtype=torch.int16, device="cuda")
    codebook = torch.randint(0x7FFFFFFFFFFFFFFF, (256,), dtype=torch.int64, device="cuda")

    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)
    # start_event.record()
    x = quiptools_cuda.decode_matmul_e8p(x, Qidxs - 0x8000, codebook)
    # end_event.record()
    # torch.cuda.synchronize()
    # elapsed_time_ms = start_event.elapsed_time(end_event)
    # print(f"Elapsed: {elapsed_time_ms:.4f}ms")


if __name__ == "__main__":
    benchmark()
