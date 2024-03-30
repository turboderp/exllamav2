
#define WARPS 32

__global__ void softmax16_topk_norm_kernel
(
    half* __restrict__ x,
    const int rows,
    const int topk
)
{
    int row = blockIdx.y * WARPS + threadIdx.x;
    if (row >= rows) return;

    // Softmax

    int4* row_ptr = (int4*) (x + row * 16);
    int4 logits_int4_a = row_ptr[0];
    int4 logits_int4_b = row_ptr[1];
    half2_uint32 l01_a(logits_int4_a.x);
    half2_uint32 l23_a(logits_int4_a.y);
    half2_uint32 l45_a(logits_int4_a.z);
    half2_uint32 l67_a(logits_int4_a.w);
    half2_uint32 l01_b(logits_int4_b.x);
    half2_uint32 l23_b(logits_int4_b.y);
    half2_uint32 l45_b(logits_int4_b.z);
    half2_uint32 l67_b(logits_int4_b.w);
    float f[] =
    {
        __low2float(l01_a.as_half2),
        __high2float(l01_a.as_half2),
        __low2float(l23_a.as_half2),
        __high2float(l23_a.as_half2),
        __low2float(l45_a.as_half2),
        __high2float(l45_a.as_half2),
        __low2float(l67_a.as_half2),
        __high2float(l67_a.as_half2),
        __low2float(l01_b.as_half2),
        __high2float(l01_b.as_half2),
        __low2float(l23_b.as_half2),
        __high2float(l23_b.as_half2),
        __low2float(l45_b.as_half2),
        __high2float(l45_b.as_half2),
        __low2float(l67_b.as_half2),
        __high2float(l67_b.as_half2)
    };

    float maxf1_a = fmaxf(f[0], f[1]);
    float maxf2_a = fmaxf(f[2], f[3]);
    float maxf3_a = fmaxf(f[4], f[5]);
    float maxf4_a = fmaxf(f[6], f[7]);
    float maxf1_b = fmaxf(f[8], f[9]);
    float maxf2_b = fmaxf(f[10], f[11]);
    float maxf3_b = fmaxf(f[12], f[13]);
    float maxf4_b = fmaxf(f[14], f[15]);
    float maxf1 = fmaxf(maxf1_a, maxf1_b);
    float maxf2 = fmaxf(maxf2_a, maxf2_b);
    float maxf3 = fmaxf(maxf3_a, maxf3_b);
    float maxf4 = fmaxf(maxf4_a, maxf4_b);
    maxf1 = fmaxf(maxf1, maxf2);
    maxf2 = fmaxf(maxf3, maxf4);
    maxf1 = fmaxf(maxf1, maxf2);

    float sum = 0;
    for (int i = 0; i < 16; ++i)
    {
        float e = expf(f[i] - maxf1);
        sum += e;
        f[i] = e;
    }
    float epsilon = 1e-8;
    float isum = 1.0f / (sum + 16 * epsilon);
    for (int i = 0; i < 16; ++i) f[i] = f[i] * isum + epsilon;

    // This is awful but surely faster than synchronizing or launching more kernels (??)

    sum = 1.0f;
    for (int i = 0; i < 16 - topk; ++i)
    {
        float minf = 1.0f;
        int minj = -1;
        for (int j = 0; j < 16; ++j)
        {
            if (f[j] > 0 && f[j] < minf)
            {
                minf = f[j];
                minj = j;
            }
        }
        sum -= f[minj];
        f[minj] = 0.0f;
    }

    __syncthreads();

    isum = 1.0f / sum;
    for (int i = 0; i < 16; ++i) f[i] *= isum;

    l01_a.as_half2 = __floats2half2_rn(f[0], f[1]);
    l23_a.as_half2 = __floats2half2_rn(f[2], f[3]);
    l45_a.as_half2 = __floats2half2_rn(f[4], f[5]);
    l67_a.as_half2 = __floats2half2_rn(f[6], f[7]);
    l01_b.as_half2 = __floats2half2_rn(f[8], f[9]);
    l23_b.as_half2 = __floats2half2_rn(f[10], f[11]);
    l45_b.as_half2 = __floats2half2_rn(f[12], f[13]);
    l67_b.as_half2 = __floats2half2_rn(f[14], f[15]);
    logits_int4_a.x = l01_a.as_uint32;
    logits_int4_a.y = l23_a.as_uint32;
    logits_int4_a.z = l45_a.as_uint32;
    logits_int4_a.w = l67_a.as_uint32;
    logits_int4_b.x = l01_b.as_uint32;
    logits_int4_b.y = l23_b.as_uint32;
    logits_int4_b.z = l45_b.as_uint32;
    logits_int4_b.w = l67_b.as_uint32;
    row_ptr[0] = logits_int4_a;
    row_ptr[1] = logits_int4_b;
}

__global__ void softmax8_topk_norm_kernel
(
    half* __restrict__ x,
    const int rows,
    const int topk
)
{
    int row = blockIdx.y * WARPS + threadIdx.x;
    if (row >= rows) return;

    // Softmax

    int4* row_ptr = (int4*) (x + row * 8);
    int4 logits_int4 = *row_ptr;
    half2_uint32 l01(logits_int4.x);
    half2_uint32 l23(logits_int4.y);
    half2_uint32 l45(logits_int4.z);
    half2_uint32 l67(logits_int4.w);
    float f[] =
    {
        __low2float(l01.as_half2),
        __high2float(l01.as_half2),
        __low2float(l23.as_half2),
        __high2float(l23.as_half2),
        __low2float(l45.as_half2),
        __high2float(l45.as_half2),
        __low2float(l67.as_half2),
        __high2float(l67.as_half2)
    };

    float maxf1 = fmaxf(f[0], f[1]);
    float maxf2 = fmaxf(f[2], f[3]);
    float maxf3 = fmaxf(f[4], f[5]);
    float maxf4 = fmaxf(f[6], f[7]);
    maxf1 = fmaxf(maxf1, maxf2);
    maxf2 = fmaxf(maxf3, maxf4);
    maxf1 = fmaxf(maxf1, maxf2);

    float sum = 0;
    for (int i = 0; i < 8; ++i)
    {
        float e = expf(f[i] - maxf1);
        sum += e;
        f[i] = e;
    }
    float epsilon = 1e-8;
    float isum = 1.0f / (sum + 8 * epsilon);
    for (int i = 0; i < 8; ++i) f[i] = f[i] * isum + epsilon;

    // This is awful but surely faster than synchronizing or launching more kernels (??)

    sum = 1.0f;
    for (int i = 0; i < 8 - topk; ++i)
    {
        float minf = 1.0f;
        int minj = -1;
        for (int j = 0; j < 8; ++j)
        {
            if (f[j] > 0 && f[j] < minf)
            {
                minf = f[j];
                minj = j;
            }
        }
        sum -= f[minj];
        f[minj] = 0.0f;
    }

    __syncthreads();

    isum = 1.0f / sum;
    for (int i = 0; i < 8; ++i) f[i] *= isum;

    l01.as_half2 = __floats2half2_rn(f[0], f[1]);
    l23.as_half2 = __floats2half2_rn(f[2], f[3]);
    l45.as_half2 = __floats2half2_rn(f[4], f[5]);
    l67.as_half2 = __floats2half2_rn(f[6], f[7]);
    logits_int4.x = l01.as_uint32;
    logits_int4.y = l23.as_uint32;
    logits_int4.z = l45.as_uint32;
    logits_int4.w = l67.as_uint32;
    *row_ptr = logits_int4;
}

__global__ void softmax4_topk_norm_kernel
(
    half* __restrict__ x,
    const int rows,
    const int topk
)
{
    int row = blockIdx.y * WARPS + threadIdx.x;
    if (row >= rows) return;

    // Softmax

    int2* row_ptr = (int2*) (x + row * 4);
    int2 logits_int2 = *row_ptr;
    half2_uint32 l01(logits_int2.x);
    half2_uint32 l23(logits_int2.y);
    float f[] =
    {
        __low2float(l01.as_half2),
        __high2float(l01.as_half2),
        __low2float(l23.as_half2),
        __high2float(l23.as_half2),
    };

    float maxf1 = fmaxf(f[0], f[1]);
    float maxf2 = fmaxf(f[2], f[3]);
    maxf1 = fmaxf(maxf1, maxf2);

    float sum = 0;
    for (int i = 0; i < 4; ++i)
    {
        float e = expf(f[i] - maxf1);
        sum += e;
        f[i] = e;
    }
    float epsilon = 1e-8;
    float isum = 1.0f / (sum + 4 * epsilon);
    for (int i = 0; i < 4; ++i) f[i] = f[i] * isum + epsilon;

    // This is awful but surely faster than synchronizing or launching more kernels (??)

    sum = 1.0f;
    for (int i = 0; i < 4 - topk; ++i)
    {
        float minf = 1.0f;
        int minj = -1;
        for (int j = 0; j < 4; ++j)
        {
            if (f[j] > 0 && f[j] < minf)
            {
                minf = f[j];
                minj = j;
            }
        }
        sum -= f[minj];
        f[minj] = 0.0f;
    }

    __syncthreads();

    isum = 1.0f / sum;
    for (int i = 0; i < 4; ++i) f[i] *= isum;

    l01.as_half2 = __floats2half2_rn(f[0], f[1]);
    l23.as_half2 = __floats2half2_rn(f[2], f[3]);
    logits_int2.x = l01.as_uint32;
    logits_int2.y = l23.as_uint32;
    *row_ptr = logits_int2;
}
