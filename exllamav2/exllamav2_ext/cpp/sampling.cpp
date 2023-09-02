#include "sampling.h"
#include "util.h"
#include <math.h>
#include <chrono>

bool* g_rep_mask = NULL;
int g_vocab_size = 0;

void apply_rep_penalty_cpu
(
    const int vocab_size,
    const uint64_t* sequence,
    const float penalty_max,
    const int sustain,
    const int decay,
    const int seq_len,
    float* logits
)
{
    if (vocab_size != g_vocab_size)
    {
        if (g_rep_mask) free(g_rep_mask);
        g_vocab_size = vocab_size;
        g_rep_mask = (bool*) malloc(g_vocab_size * sizeof(bool));
    }

    memset(g_rep_mask, 0, g_vocab_size * sizeof(bool));

    float v = penalty_max;
    float dv = decay ? (1.0f - penalty_max) / (float) decay : 0.0f;

    int s = sustain == -1 ? seq_len : sustain;
    int beg = seq_len - s - decay;
    if (beg < 0) beg = 0;

    for (int i = seq_len; i > beg;)
    {
        uint64_t t = sequence[--i];
        if (!g_rep_mask[t])
        {
            if (logits[t] > 0.0) logits[t] /= v;
            else logits[t] *= v;
            g_rep_mask[t] = true;
        }
        if (--s < 0) v += dv;
    }
}

void softmax_cpu
(
    const int vocab_size,
    const float temperature,
    const float* logits,
    float* output
)
{
    float esum = 0.0f;
    float itemp = 1.0f / temperature;
    #pragma unroll(32)
    for (int i = 0; i < vocab_size; i++) { float e = expf(logits[i] * itemp); output[i] = e; esum += e; }
    float isum = 1.0f / esum;
    #pragma unroll(32)
    for (int i = 0; i < vocab_size; i++) { output[i] *= isum; }
}

void normalize_cpu
(
    const int num_candidates,
    float* probs
)
{
    float sum = 0.0f;
    #pragma unroll(32)
    for (int i = 0; i < num_candidates; i++) sum += probs[i];
    float isum = 1.0f / sum;
    #pragma unroll(32)
    for (int i = 0; i < num_candidates; i++) probs[i] *= isum;
}

int greedy_sample
(
    const int num_candidates,
    const float* probs
)
{
    int maxidx = 0;
    float max = probs[0];

    for(int i = 1; i < num_candidates; i++)
    {
        if (probs[i] > max)
        {
            max = probs[i];
            maxidx = i;
        }
    }
    return maxidx;
}

template <typename T>
inline void swap(T &a, T &b)
{
    T temp = a;
    a = b;
    b = temp;
}

void quicksort_with_idx_desc
(
    float* arr,
    int* idx,
    int low,
    int high,
    int max_index
)
{
	if (low >= high) return;

    // Bubblesort very short segments

    if (high - low == 1)
    {
        int i0 = low;
        int i1 = low + 1;

        if (arr[i0] < arr[i1]) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        return;
    }

    if (high - low == 2)
    {
        int i0 = low;
        int i1 = low + 1;
        int i2 = low + 2;

        if (arr[i0] < arr[i1]) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (arr[i1] < arr[i2]) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (arr[i0] < arr[i1]) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        return;
    }

    if (high - low == 3)
    {
        int i0 = low;
        int i1 = low + 1;
        int i2 = low + 2;
        int i3 = low + 3;

        if (arr[i0] < arr[i1]) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (arr[i1] < arr[i2]) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (arr[i2] < arr[i3]) { swap<float>(arr[i2], arr[i3]); swap<int>(idx[i2], idx[i3]); }
        if (arr[i0] < arr[i1]) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (arr[i1] < arr[i2]) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (arr[i0] < arr[i1]) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        return;
    }

    if (high - low == 4)
    {
        int i0 = low;
        int i1 = low + 1;
        int i2 = low + 2;
        int i3 = low + 3;
        int i4 = low + 4;

        if (arr[i0] < arr[i1]) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (arr[i1] < arr[i2]) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (arr[i2] < arr[i3]) { swap<float>(arr[i2], arr[i3]); swap<int>(idx[i2], idx[i3]); }
        if (arr[i3] < arr[i4]) { swap<float>(arr[i3], arr[i4]); swap<int>(idx[i3], idx[i4]); }
        if (arr[i0] < arr[i1]) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (arr[i1] < arr[i2]) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (arr[i2] < arr[i3]) { swap<float>(arr[i2], arr[i3]); swap<int>(idx[i2], idx[i3]); }
        if (arr[i0] < arr[i1]) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (arr[i1] < arr[i2]) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (arr[i0] < arr[i1]) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        return;
    }

    // Quicksort longer segments

	float pivot = arr[high];
	int i = low;
	int j = low;
    #pragma unroll(4)
	while(i <= high)
	{
		if(arr[i] < pivot) i++;
		else { swap<float>(arr[i], arr[j]); swap<int>(idx[i], idx[j]); i++; j++; }
	}

	int pos = j - 1;

    // We know in advance we'll only need max_index elements of the sorted array so skip sorting segments after that

	if (max_index == 0 || max_index >= low)
	    quicksort_with_idx_desc(arr, idx, low, pos - 1, max_index);
	if (max_index == 0 || max_index > pos)
	    quicksort_with_idx_desc(arr, idx, pos + 1, high, max_index);
}

void sort_descending
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    int max_index
)
{
//    auto start = std::chrono::high_resolution_clock::now();
    quicksort_with_idx_desc(temp_probs, temp_indices, 0, num_candidates - 1, max_index);
//    auto stop = std::chrono::high_resolution_clock::now();
//    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//    DBGI(duration_us);

    //for (int i = 0; i < (max_index == 0 ? num_candidates : max_index) - 1; i++) if (temp_probs[i] < temp_probs[i + 1]) DBGI(i);
}

int top_k_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    int top_k
)
{
    // TODO: Currently relies on sorting the logits with early exit. Heap would probably be faster
    // TODO: Selection sort should be faster for very low values of K

    return top_k;
}

int top_p_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float top_p
)
{
    // TODO: Maybe special case when top-K is disabled: select max until sum exceeds P, for P < ~0.75

    float cumprob = 0.0f;
    int num = 0;

    while (true)
    {
        cumprob += temp_probs[num];
        if (cumprob >= top_p) break;
        num++;
        if (num >= num_candidates) break;
    }

    if (num == 0) num = 1;
    return num;
}

int multinomial_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float random
)
{
    int idx = 0;
    float accum = temp_probs[idx];

    while (true)
    {
        if (accum >= random) break;
        if (idx == num_candidates - 1) break;
        idx++;
        accum += temp_probs[idx];
    }

    temp_probs[0] = temp_probs[idx];
    temp_indices[0] = temp_indices[idx];

    return 1;
}




