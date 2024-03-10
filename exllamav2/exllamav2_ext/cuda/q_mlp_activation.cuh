
__device__ __forceinline__ half silu(half x)
{
    half one = __float2half(1.0f);
    half neg_x = __hneg(x);
    half e = hexp(neg_x);
    half sum = __hadd(one, e);
    half r = hrcp(sum);
    half result = __hmul(x, r);
    return result;
}

__device__ __forceinline__ half2 silu(half2 x)
{
    half2 one = __float2half2_rn(1.0f);
    half2 neg_x = __hneg2(x);
    half2 e = h2exp(neg_x);
    half2 sum = __hadd2(one, e);
    half2 r = h2rcp(sum);
    half2 result = __hmul2(x, r);
    return result;
}

__device__ __forceinline__ half gelu(half x)
{
//    float xf = __half2float(x);
//    const float cdf = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (xf + 0.044715f * xf * xf * xf))));
//    return __float2half_rn(xf * cdf);
//
    float xf = __half2float(x);
    const float c = 0.797884560803f;  // sqrt(2/Pi)
    float tanh_arg = c * (xf + 0.044715f * xf * xf * xf);
    xf = 0.5f * xf * (1.0 + tanh_opt(tanh_arg));
    return __float2half_rn(xf);
}

__device__ __forceinline__ half2 gelu(half2 x)
{
    return __halves2half2(gelu(__low2half(x)), gelu(__high2half(x)));
}

// Activation with gate

typedef void (*fp_act_mul_kernel)
(
    half*,
    const half*,
    const int,
    const int,
    const half*,
    const int
);

template <bool use_half2, bool use_r_weights, bool act_fn_gelu>
__global__ void act_mul_kernel
(
    half* __restrict__ x,
    const half* __restrict__ y,
    const int height,
    const int width,
    const half* r_weights,
    const int r_weights_stride
)
{
    MatrixView_half_rw x_(x, height, width);
    MatrixView_half y_(y, height, width);

    int column = (THREADS_X * blockIdx.x + threadIdx.x); if constexpr (use_half2) column *= 2;
    int row = THREADS_Y * blockIdx.y + threadIdx.y;
    if (row >= height) return;

    if constexpr (use_r_weights)
    {
        half_uint16 weight(r_weights[row * r_weights_stride]);
        if (!weight.as_uint16)
        {
//             half2 ppp = __float2half2_rn(6.9f);
//             x_.set_half2(row, column, ppp);
            return;
        }
    }

    // act(x) * y

    if constexpr (use_half2)
    {
        half2 x_item = x_.item_half2(row, column);
        half2 y_item = y_.item_half2(row, column);

        if constexpr (act_fn_gelu)
            x_item = gelu(x_item);
        else
            x_item = silu(x_item);

        x_item = __hmul2(x_item, y_item);

        x_.set_half2(row, column, x_item);
    }
    else
    {
        half x_item = x_.item(row, column);
        half y_item = y_.item(row, column);

        if constexpr (act_fn_gelu)
            x_item = gelu(x_item);
        else
            x_item = silu(x_item);

        x_item = __hmul(x_item, y_item);

        x_.set(row, column, x_item);
    }
}

fp_act_mul_kernel pick_act_mul_kernel(bool use_half2, bool mul_r_weights, bool act_fn_gelu)
{
    if (act_fn_gelu)
    {
        if ( use_half2 && !mul_r_weights) return act_mul_kernel< true, false,  true>;
        if ( use_half2 &&  mul_r_weights) return act_mul_kernel< true,  true,  true>;
        if (!use_half2 && !mul_r_weights) return act_mul_kernel<false, false,  true>;
        if (!use_half2 &&  mul_r_weights) return act_mul_kernel<false,  true,  true>;
    }
    else
    {
        if ( use_half2 && !mul_r_weights) return act_mul_kernel< true, false, false>;
        if ( use_half2 &&  mul_r_weights) return act_mul_kernel< true,  true, false>;
        if (!use_half2 && !mul_r_weights) return act_mul_kernel<false, false, false>;
        if (!use_half2 &&  mul_r_weights) return act_mul_kernel<false,  true, false>;
    }
    return NULL;
};

// Activation without gate

typedef void (*fp_act_kernel)
(
    half*,
    const int,
    const int,
    const half*,
    const int
);

template <bool use_half2, bool use_r_weights, bool act_fn_gelu>
__global__ void act_kernel
(
    half* __restrict__ x,
    const int height,
    const int width,
    const half* r_weights,
    const int r_weights_stride
)
{
    MatrixView_half_rw x_(x, height, width);

    int column = (THREADS_X * blockIdx.x + threadIdx.x); if constexpr (use_half2) column *= 2;
    int row = THREADS_Y * blockIdx.y + threadIdx.y;
    if (row >= height) return;

    if constexpr (use_r_weights)
    {
        half_uint16 weight(r_weights[row * r_weights_stride]);
        if (!weight.as_uint16)
        {
//             half2 ppp = __float2half2_rn(6.9f);
//             x_.set_half2(row, column, ppp);
            return;
        }
    }

    // act(x) * y

    if constexpr (use_half2)
    {
        half2 x_item = x_.item_half2(row, column);

        if constexpr (act_fn_gelu)
            x_item = gelu(x_item);
        else
            x_item = silu(x_item);

        x_.set_half2(row, column, x_item);
    }
    else
    {
        half x_item = x_.item(row, column);

        if constexpr (act_fn_gelu)
            x_item = gelu(x_item);
        else
            x_item = silu(x_item);

        x_.set(row, column, x_item);
    }
}

fp_act_kernel pick_act_kernel(bool use_half2, bool mul_r_weights, bool act_fn_gelu)
{
    if (act_fn_gelu)
    {
        if ( use_half2 && !mul_r_weights) return act_kernel< true, false,  true>;
        if ( use_half2 &&  mul_r_weights) return act_kernel< true,  true,  true>;
        if (!use_half2 && !mul_r_weights) return act_kernel<false, false,  true>;
        if (!use_half2 &&  mul_r_weights) return act_kernel<false,  true,  true>;
    }
    else
    {
        if ( use_half2 && !mul_r_weights) return act_kernel< true, false, false>;
        if ( use_half2 &&  mul_r_weights) return act_kernel< true,  true, false>;
        if (!use_half2 && !mul_r_weights) return act_kernel<false, false, false>;
        if (!use_half2 &&  mul_r_weights) return act_kernel<false,  true, false>;
    }
    return NULL;
};


