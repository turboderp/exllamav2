
import torch
import quiptools_cuda
import time
torch.manual_seed(0)
m = 8192*2
n = 8192//2

cb = torch.randn(256,8,dtype=torch.float16,device="cuda")
cb_even = cb.sum(dim=-1) > 0
yidxs = torch.randint(2**16,(m,n//8),device="cuda").to(torch.int16)
y1 = torch.zeros(m,n,dtype=torch.float16,device="cuda")

'''
torch.cuda.synchronize()
start = time.time()
y = cb[yidxs.view(-1).to(torch.int32)+2**15,:].view(m,n)
torch.cuda.synchronize()
end = time.time()
print(f"elapsed for pure torch: {end - start}")
'''

torch.cuda.synchronize()
start = time.time()
quiptools_cuda.decompress_e8p_origorder(yidxs,cb,cb_even,y1)
torch.cuda.synchronize()
end = time.time()
print(f"elapsed for orig decompress_e8p: {end - start}")

print(y1)

#assert((y1 == y).all())
