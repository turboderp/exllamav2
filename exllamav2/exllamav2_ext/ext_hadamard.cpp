#include <torch/extension.h>
#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "config.h"
#include "cpp/util.h"

#define HALF_P 0x3C00
#define HALF_N 0xBC00
#define HALF_PP 0x3C003C00
#define HALF_PN 0xBC003C00
#define HALF_NP 0x3C00BC00
#define HALF_NN 0xBC00BC00

inline int pmod(int a, int b)
{
    int ret = a % b;
    if (ret < 0 && b > 0) ret += b;
    return ret;
}

inline int modular_pow(int base, int exp, int mod)
{
    int result = 1;
    base = pmod(base, mod);
    while (exp > 0)
    {
        if (exp % 2 == 1) result = pmod((result * base), mod);
        exp = exp >> 1;
        base = pmod((base * base), mod);
    }
    return result;
}

inline bool is_quadratic_residue(int a, int p)
{
    return modular_pow(a, (p - 1) / 2, p) == 1;
}

// Paley construction

void had_paley
(
    torch::Tensor h
)
{
    TORCH_CHECK_DTYPE(h, kHalf);
    TORCH_CHECK_SHAPES(h, 0, h, 1, 1);
    TORCH_CHECK(h.is_contiguous());
    int n = h.size(0);
    int p = n - 1;
    uint16_t* ptr = (uint16_t*) h.data_ptr();

    for (int j = 0; j < n; ++j)
        *ptr++ = HALF_P;

    for (int i = 0; i < p; ++i)
    {
        *ptr++ = HALF_N;
        for (int j = 0; j < p; ++j)
        {
            if (i == j) *ptr++ = HALF_P;
            else
            {
                int residue = pmod(i - j, p);
                if (is_quadratic_residue(residue, p))
                    *ptr++ = HALF_P;
                else
                    *ptr++ = HALF_N;
            }
        }
    }
}

// Paley construction, type 2

void had_paley2
(
    torch::Tensor h
)
{
    TORCH_CHECK_DTYPE(h, kHalf);
    TORCH_CHECK_SHAPES(h, 0, h, 1, 1);
    int n = h.size(0);
    int p = n / 2 - 1;
    uint32_t* ptr0 = (uint32_t*) h.data_ptr();
    uint32_t* ptr1 = ptr0 + n / 2;

    for (int i = 0; i < n / 2; ++i)
    {
        for (int j = 0; j < n / 2; ++j)
        {
            if (i == j)
            {
                *ptr0++ = HALF_PN;
                *ptr1++ = HALF_NN;
            }
            else
            {
                int residue = pmod(i - j, p);
                if (i == 0 || j == 0 || is_quadratic_residue(residue, p))
                {
                    *ptr0++ = HALF_PP;
                    *ptr1++ = HALF_PN;
                }
                else
                {
                    *ptr0++ = HALF_NN;
                    *ptr1++ = HALF_NP;
                }
            }
        }
        ptr0 += n / 2;
        ptr1 += n / 2;
    }
}
