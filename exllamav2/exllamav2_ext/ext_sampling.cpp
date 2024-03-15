#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef __linux__
#include <mm_malloc.h>
#else
#define _mm_malloc(a, b) _aligned_malloc(a, b)
#define _mm_free(a) _aligned_free(a)
#endif

#include "config.h"
#include "ext_sampling.h"

#include "cpp/sampling.h"

#include "cpp/util.h"

void apply_rep_penalty
(
    torch::Tensor sequence,
    float penalty_max,
    int sustain,
    int decay,
    float alpha_frequency,
    float alpha_presence,
    torch::Tensor logits
)
{
    TORCH_CHECK_DTYPE(sequence, kLong);
    TORCH_CHECK_DTYPE(logits, kFloat);
    TORCH_CHECK_SHAPES(sequence, 0, logits, 0, 1);

    int vocab_size = logits.size(-1);
    int bsz = sequence.size(0);
    int seq_len = sequence.size(-1);

    for (int i = 0; i < bsz; i++)
    {
        apply_rep_penalty_cpu
        (
            vocab_size,
            ((uint64_t*) sequence.data_ptr()) + i * seq_len,
            penalty_max,
            sustain,
            decay,
            alpha_frequency,
            alpha_presence,
            seq_len,
            ((float*) logits.data_ptr()) + i * vocab_size
        );
    }
}

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
    float min_temp = 0,
    float max_temp = 0.0f,
    float temp_exponent = 1.0f,
    float smoothing_factor = 0.0f,
    float skew = 0.0f
)
{
    TORCH_CHECK_DTYPE(logits, kFloat);
    TORCH_CHECK_DTYPE(output_tokens, kLong);
    TORCH_CHECK_DTYPE(output_probs, kFloat);
    TORCH_CHECK_DTYPE(logits, kFloat);
    TORCH_CHECK_DTYPE(logit_filter, kBool);

    TORCH_CHECK_SHAPES(logit_filter, 0, logits, 0, 1);
    TORCH_CHECK_SHAPES(logit_filter, 1, logits, 1, 1);

    int vocab_size = logits.size(-1);
    int bsz = logits.size(0);

    float* temp_probs = (float*) _mm_malloc(vocab_size * sizeof(float), 32);
    int* temp_indices = (int*) _mm_malloc(vocab_size * sizeof(int), 32);
    float* logits_ptr = (float*) logits.data_ptr();

    int num_probs = 0;
    if (!output_kprobs.device().is_meta())
        num_probs = output_kprobs.size(2);

    bool* logits_filter_ptr = (bool*) logit_filter.data_ptr();

    if (temperature < 0.01)
    {
        temperature = 1.0f;
        top_k = 1;
    }

    for (int i = 0; i < bsz; i++)
    {
        float exponent = 1.0f;
        if (smoothing_factor > 0)
        {
            exponent = 2.0f;
            temperature /= smoothing_factor;
        }

        softmax_cpu
        (
             vocab_size,
             temperature,
             logits_ptr + i * vocab_size,
             logits_filter_ptr + i * vocab_size,
             exponent,
             temp_probs
        );

        for (int j = 0; j < vocab_size; j++) temp_indices[j] = j;
        int num_candidates = vocab_size;

        if (top_k > 0 && top_k < vocab_size)
        {
            num_candidates = top_k_cpu(num_candidates, temp_probs, temp_indices, top_k);
            normalize_cpu(num_candidates, temp_probs);
        }

        if (num_candidates > 1 && top_p > 0.0f && top_p < 1.0f)
        {
            num_candidates = top_p_cpu(num_candidates, temp_probs, temp_indices, top_p);
            normalize_cpu(num_candidates, temp_probs);
        }

        if (num_candidates > 1 && top_a > 0.0f)
        {
            num_candidates = top_a_cpu(num_candidates, temp_probs, temp_indices, top_a);
            normalize_cpu(num_candidates, temp_probs);
        }

        if (num_candidates > 1 && min_p > 0.0f && min_p < 1.0f)
        {
            num_candidates = min_p_cpu(num_candidates, temp_probs, temp_indices, min_p);
            normalize_cpu(num_candidates, temp_probs);
        }

        if (num_candidates > 1 && tfs > 0.0f && tfs < 1.0f)
        {
            num_candidates = tfs_cpu(num_candidates, temp_probs, temp_indices, tfs);
            normalize_cpu(num_candidates, temp_probs);
        }

        if (num_candidates > 1 && typical > 0.0f && typical < 1.0f)
        {
            num_candidates = typical_cpu(num_candidates, temp_probs, temp_indices, typical);
            normalize_cpu(num_candidates, temp_probs);
        }

        if (num_candidates > 1 && mirostat)
        {
            num_candidates = mirostat_pre_cpu(num_candidates, temp_probs, temp_indices, mirostat_mu[i], mirostat_tau, mirostat_eta);
            normalize_cpu(num_candidates, temp_probs);
        }

        if (num_candidates > 1 && (post_temperature != 1.0f || max_temp > min_temp))
        {
            num_candidates = post_softmax_temperature(num_candidates, temp_probs, temp_indices, post_temperature, min_temp, max_temp, temp_exponent);
        }

        if (num_probs || (skew != 0.0f))
        {
            num_candidates = pre_sort_descending(num_candidates, temp_probs, temp_indices);
            sort_descending(num_candidates, temp_probs, temp_indices, num_probs);
        }

        if (num_probs)
        {
            int j = 0;
            for (; j < num_candidates && j < num_probs; ++j)
            {
                float tp = temp_probs[j];
                if (tp == 0.0f) break;

                output_ktokens[i][0][j] = temp_indices[j];
                output_kprobs[i][0][j] = tp;
            }

            // Candidate tokens are only valid up to num_candidates, so fake the ones with prob == 0

            int fake_idx = 0;
            for (; j < num_probs; ++j)
            {
                for (int k = 0; k < num_candidates; ++k)
                    if (temp_indices[k] == fake_idx) { fake_idx++; k = 0; }

                output_ktokens[i][0][j] = fake_idx;
                output_kprobs[i][0][j] = 0.0f;
                fake_idx++;
            }
        }

        float random_s = random;

        if (skew != 0.0f)
        {
            random_s = powf(random, expf(-skew));
        }

        multinomial_cpu(num_candidates, temp_probs, temp_indices, random_s);

        output_tokens[i][0] = temp_indices[0];
        output_probs[i][0] = temp_probs[0];

        if (mirostat)
        {
            mirostat_mu[i] = mirostat_post_cpu(num_candidates, temp_probs, temp_indices, mirostat_mu[i], mirostat_tau, mirostat_eta);
        }

        // Derive some more totally random numbers for subsequent samples in the same batch

        if (bsz > 1)
        {
            float r = random;
            for (int j = 0; j < 10; ++j)
            {
                r += 1.337 + random;
                r *= r;
                r = fmod(r, 1.0f);
            }
            random = r;
        }
    }

    _mm_free(temp_probs);
    _mm_free(temp_indices);

    return mirostat_mu;
}

void logit_filter_exclusive
(
    torch::Tensor filter,                                       // shape [bsz, vocab_size]
    const std::vector<std::vector<int>> &exclusive_lists
)
{
    TORCH_CHECK_DTYPE(filter, kBool);
    TORCH_CHECK((uint64_t) filter.size(0) == exclusive_lists.size(), "Number of lists does not match batch size")

    bool* filter_ptr = (bool*) filter.data_ptr();
    unsigned int vocab_size = filter.size(1);

    for(const auto& list : exclusive_lists)
    {
        unsigned int id = 0;
        unsigned int next_id_idx = 0;
        unsigned int next_id = list[next_id_idx];

        while (id < vocab_size)
        {
            while (id < next_id)
            {
                filter_ptr[id] = false;
                id++;
            }
            id++;
            next_id_idx++;
            if (next_id_idx >= list.size()) next_id = vocab_size;
            else next_id = list[next_id_idx];
        }

        filter_ptr += vocab_size;
    }
}

void fast_fill_cpu_ones_bool(torch::Tensor tensor)
{
    TORCH_CHECK_DTYPE(tensor, kBool);
    memset(tensor.data_ptr(), 1, tensor.numel());
}

void fast_fadd_cpu(torch::Tensor a, torch::Tensor b)
{
    TORCH_CHECK_DTYPE(a, kFloat);
    TORCH_CHECK_DTYPE(b, kFloat);
    int n = a.numel();
    int m = b.numel();
    int bsz = n / m;
    TORCH_CHECK(bsz * m == n, "a and b are incompatible sizes");

    float* a_ptr = (float*) a.data_ptr();
    float* b_ptr = (float*) b.data_ptr();

    for (int i = 0; i < bsz; ++i)
    {
        float* b_ptr_ = b_ptr;
        for (int j = 0; j < m; ++j)
            *a_ptr++ += *b_ptr_++;
    }
}

void fast_copy_cpu(torch::Tensor a, torch::Tensor b)
{
    size_t size_a = a.numel() * torch::elementSize(torch::typeMetaToScalarType(a.dtype()));
    size_t size_b = b.numel() * torch::elementSize(torch::typeMetaToScalarType(b.dtype()));
    TORCH_CHECK(size_a == size_b, "a and b are not the same size");
    memcpy(a.data_ptr(), b.data_ptr(), size_a);
}

void dump_profile_results()
{
    profile_results();
}