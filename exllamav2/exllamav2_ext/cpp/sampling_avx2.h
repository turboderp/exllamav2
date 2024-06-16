#ifndef _sampling_avx2_h
#define _sampling_avx2_h

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <string>

int softmax_cpu_avx2
(
    const int vocab_size,
    const float temperature,
    const float* logits,
    const bool* logits_filter,
    const float exponent,
    float* output
);

#endif