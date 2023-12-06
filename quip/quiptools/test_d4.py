
import torch
import quiptools_cuda
import time

k = 32*32
m = 8192*2
n = 8192//2

x = torch.randn(k,n,dtype=torch.float16,device="cuda")
z = torch.zeros(k,m,dtype=torch.float16,device="cuda")
cb = torch.randn(256,4,dtype=torch.float16,device="cuda")
yidxs = torch.randint(256,(m,n//4),device="cuda").to(torch.uint8)

yidxs_reordered = yidxs.view(m,n//(4*4),4).permute(1,0,2).reshape(m,n//4).contiguous()

y1 = torch.zeros(m,n,dtype=torch.float16,device="cuda")

# yidxs_reordered = yidxs.view(m//32,32,n//(4*4),4).permute(0,2,1,3).reshape(m,n//4).contiguous()

# yidxs_reordered_k16 = yidxs.view(m//16,16,n//(4*4),4).permute(0,2,1,3).reshape(m,n//4).contiguous()

torch.cuda.synchronize()
start = time.time()
y = cb[yidxs.view(-1).to(torch.int32),:].view(m,n)
z0 = x @ y.t()
torch.cuda.synchronize()
end = time.time()
print(f"elapsed for pure torch: {end - start}")


torch.cuda.synchronize()
start = time.time()
quiptools_cuda.decompress_d4_origorder(yidxs,cb,y1)
torch.cuda.synchronize()
end = time.time()
print(f"elapsed for orig decompress_d4: {end - start}")

assert((y1 == y).all())

y1.zero_()
torch.cuda.synchronize()
start = time.time()
quiptools_cuda.decompress_d4(yidxs_reordered,cb,y1)
z1 = x @ y1.t()
torch.cuda.synchronize()
end = time.time()
print(f"elapsed for decompress_d4 and multiply: {end - start}")

assert((y1 == y).all())


torch.cuda.synchronize()
start = time.time()
quiptools_cuda.lookupmatmul_d4_k8(x,yidxs_reordered,cb,z)
torch.cuda.synchronize()
end = time.time()
print(f"   elapsed for k8 cuda: {end - start}")

lookupmatmul_d4_k8_err = ((z.to(torch.float32) - z0.to(torch.float32)).square().sum() / ((z0.to(torch.float32)).square().sum()+1e-10))
print(f"lookupmatmul_d4_k8 error: {lookupmatmul_d4_k8_err}")

torch.cuda.synchronize()
start = time.time()
quiptools_cuda.lookupmatmul_d4_k16(x,yidxs_reordered,cb,z)
torch.cuda.synchronize()
end = time.time()
print(f"  elapsed for k16 cuda: {end - start}")


lookupmatmul_d4_k16_err = ((z.to(torch.float32) - z0.to(torch.float32)).square().sum() / ((z0.to(torch.float32)).square().sum()+1e-10))
print(f"lookupmatmul_d4_k16 error: {lookupmatmul_d4_k16_err}")


torch.cuda.synchronize()
start = time.time()
quiptools_cuda.lookupmatmul_d4_k32(x,yidxs_reordered,cb,z)
torch.cuda.synchronize()
end = time.time()
print(f"  elapsed for k32 cuda: {end - start}")


lookupmatmul_d4_k32_err = ((z.to(torch.float32) - z0.to(torch.float32)).square().sum() / ((z0.to(torch.float32)).square().sum()+1e-10))
print(f"lookupmatmul_d4_k32 error: {lookupmatmul_d4_k32_err}")
