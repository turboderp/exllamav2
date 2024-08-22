
#include "cpp/generator.h"

void apply_rep_penalty
(
    torch::Tensor sequence,
    float penalty_max,
    int sustain,
    int decay,
    float alpha_frequency,
    float alpha_presence,
    torch::Tensor logits
);

std::vector<float> sample_basic
(
    torch::Tensor logits,           // shape [bsz, vocab_size]
    float temperature,
    int top_k,
    float top_p,
    float top_a,
    float min_p,
    float tfs,
    float typical,
    float random,
    torch::Tensor output_tokens,    // shape [bsz, 1]
    torch::Tensor output_probs,     // shape [bsz, 1]
    torch::Tensor output_kprobs,    // None or [bsz, 1, num_probs]
    torch::Tensor output_ktokens,   // None or [bsz, 1, num_probs]
    torch::Tensor logit_filter,     // shape [bsz, vocab_size]
    bool mirostat,
    std::vector<float>& mirostat_mu,
    float mirostat_tau,
    float mirostat_eta,
    float post_temperature,
    float min_temp,
    float max_temp,
    float temp_exponent,
    float smoothing_factor,
    float skew
);

void logit_filter_exclusive
(
    torch::Tensor filter,                                       // shape [bsz, vocab_size]
    const py::list& exclusive_lists
);

void fast_fill_cpu_ones_bool(torch::Tensor tensor);

void fast_fadd_cpu(torch::Tensor a, torch::Tensor b);

void fast_copy_cpu(torch::Tensor a, torch::Tensor b);

void dump_profile_results();