#include "sampling_avx2.h"
#include "sampling.h"
#include "util.h"
#include "algorithm"
#include <math.h>
#include <vector>
#include <queue>
#include <utility>
#include <chrono>
#include <map>
#include "avx2_target.h"
#include "avx_mathfun.h"
#include "profiling.h"

AVX2_TARGET
int softmax_cpu_avx2
(
    const int vocab_size,
    const float temperature,
    const float* logits,
    const bool* logits_filter,
    const float exponent,
    float* output
)
{
    profile_start("softmax_cpu (AVX2)");

    int vocab_size_aligned = ((vocab_size + 31) / 32) * 32;
    float esum = 0.0f;
    float itemp = 1.0f / temperature;
    const float minf = -1e38;
    float maxl = minf;
    float maxi;

    // Apply logit filter and find max logit

    for (int i = 0; i < vocab_size; ++i)
    {
        float l = logits[i];
        bool f = !logits_filter || logits_filter[i];
        l = f ? l : minf;
        if (l > maxl)
        {
            maxl = l;
            maxi = i;
        }
        output[i] = l;
    }

    for (int i = vocab_size; i < vocab_size_aligned; i++)
        output[i] = minf;

    // SIMD values

    __m256 maxl8  = _mm256_set1_ps(maxl);
    __m256 itemp8 = _mm256_set1_ps(itemp);
    __m256 esum8  = _mm256_set1_ps(esum);

    // Apply temperature, exponentiate and compute exponential sum

    if (exponent == 2.0f)
    {
        __m256 sign_mask = _mm256_set1_ps(-0.0f);
        for (int i = 0; i < vocab_size_aligned; i += 8)
        {
            __m256 x = _mm256_load_ps(&output[i]);
            x = _mm256_sub_ps(x, maxl8);
            x = _mm256_mul_ps(x, x);
            x = _mm256_xor_ps(x, sign_mask);
            x = _mm256_mul_ps(x, itemp8);
            x = exp256_ps(x);
            _mm256_store_ps(&output[i], x);
            esum8 = _mm256_add_ps(esum8, x);
        }
    }
    else if (exponent != 1.0f)
    {
        for (int i = 0; i < vocab_size_aligned; i++)
        {
            float l = output[i] - maxl;
            l = -powf(fabs(l), exponent);
            float e = expf(l * itemp);
            output[i] = e;
            esum += e;
        }
    }
    else
    {
        if (itemp == 1.0f)
        {
            for (int i = 0; i < vocab_size_aligned; i += 8)
            {
                __m256 x = _mm256_load_ps(&output[i]);
                x = _mm256_sub_ps(x, maxl8);
                x = exp256_ps(x);
                _mm256_store_ps(&output[i], x);
                esum8 = _mm256_add_ps(esum8, x);
            }
        }
        else
        {
            for (int i = 0; i < vocab_size_aligned; i += 8)
            {
                __m256 x = _mm256_load_ps(&output[i]);
                x = _mm256_sub_ps(x, maxl8);
                x = _mm256_mul_ps(x, itemp8);
                x = exp256_ps(x);
                _mm256_store_ps(&output[i], x);
                esum8 = _mm256_add_ps(esum8, x);
            }
        }
    }

    // Normalize

    float xv[8];
    _mm256_store_ps(xv, esum8);
    for (int k = 0; k < 8; ++k) esum += xv[k];
    float isum = 1.0f / esum;
    __m256 isum8  = _mm256_set1_ps(isum);

    for (int i = 0; i < vocab_size_aligned; i += 8)
    {
        __m256 x = _mm256_load_ps(&output[i]);
        x = _mm256_mul_ps(x, isum8);
        _mm256_store_ps(&output[i], x);
    }

    profile_stop();

//    printf("Softmax:");
//    float summ = 0.0f;
//    for (int i = 0; i < vocab_size; i++)
//    {
//        if (logits_filter[i])
//        {
//            summ += output[i];
//            if (output[i] < 1e-5) continue;
//            printf("%d, %f\n", i, output[i]);
//        }
//    }
//    printf("sum: %f\n\n", summ);
    return maxi;
}
