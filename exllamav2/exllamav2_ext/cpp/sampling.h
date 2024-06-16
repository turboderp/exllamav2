#ifndef _sampling_h
#define _sampling_h

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <string>

void profile_results();
void profile_start(std::string stage);
void profile_stop();

void apply_rep_penalty_cpu
(
    const int vocab_size,
    const uint64_t* sequence,
    const float penalty_max,
    const int sustain,
    const int decay,
    const float alpha_frequency,
    const float alpha_presence,
    const int seq_len,
    float* logits
);

int softmax_cpu
(
    const int vocab_size,
    const float temperature,
    const float* logits,
    const bool* logits_filter,
    const float exponent,
    float* output
);

void normalize_cpu
(
    const int num_candidates,
    float* probs
);

int pre_sort_descending
(
    const int num_candidates,
    float* arr,
    int* idx
);

int sort_descending
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
    int top_k,
    int maxlogit = -1
);

int top_p_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float top_p
);

int top_a_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float top_a
);

int min_p_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float min_p
);

int tfs_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float tfs
);

int typical_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float typical
);

int mirostat_pre_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float mirostat_mu,
    float mirostat_tau,
    float mirostat_eta
);

float mirostat_post_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float mirostat_mu,
    float mirostat_tau,
    float mirostat_eta
);

int post_softmax_temperature
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float temp,
    float min_temp,
    float max_temp,
    float temp_exponent
);

int multinomial_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float random
);

#endif
