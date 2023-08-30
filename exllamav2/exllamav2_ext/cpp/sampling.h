#ifndef _sampling_h
#define _sampling_h

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdio>

void apply_rep_penalty_cpu
(
    const int vocab_size,
    const uint64_t* sequence,
    const float penalty_max,
    const int sustain,
    const int decay,
    const int seq_len,
    float* logits
);

void softmax_cpu
(
    const int vocab_size,
    const float temperature,
    const float* logits,
    float* output
);

void normalize_cpu
(
    const int num_candidates,
    float* probs
);

int greedy_sample
(
    const int num_candidates,
    const float* probs
);

void sort_descending
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    int max_index
);

int top_k_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    int top_k
);

int top_p_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float top_p
);

int multinomial_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float random
);

#endif


