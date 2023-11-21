#include "sampling.h"
#include "util.h"
#include <math.h>
#include <vector>
#include <queue>
#include <utility>

const int top_k_heap_threshold = 500;

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
    const bool* logits_filter,
    float* output
)
{
    float esum = 0.0f;
    float itemp = 1.0f / temperature;
    float maxl = 0.0f;

    #pragma unroll(32)
    for (int i = 0; i < vocab_size; i++)
    {
        if (!logits_filter[i]) continue;
        maxl = fmaxf(logits[i], maxl);
    }
    maxl *= itemp;

    #pragma unroll(32)
    for (int i = 0; i < vocab_size; i++)
    {
        if (!logits_filter[i]) continue;
        float e = expf(logits[i] * itemp - maxl);
        output[i] = e;
        esum += e;
    }
    float isum = 1.0f / esum;

    #pragma unroll(32)
    for (int i = 0; i < vocab_size; i++)
    {
        if (logits_filter[i])
            output[i] *= isum;
        else
            output[i] = 0.0f;
    }

//    printf("Softmax:");
//    float summ = 0.0f;
//    for (int i = 0; i < vocab_size; i++)
//    {
//        if (logits_filter[i])
//        {
//            printf("%d, %f\n", i, output[i]);
//            summ += output[i];
//        }
//    }
//    printf("sum: %f\n", summ);
}

int post_softmax_temperature
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float temperature
)
{
//    printf("---- pre\n");
//    for (int i = 0; i < num_candidates; ++i)
//        DBGIF(i, temp_probs[i]);

    float psum = 0.0f;
    float itemp = 1.0f / temperature;
    for (int i = 0; i < num_candidates; ++i)
    {
        float p = powf(temp_probs[i], itemp);
        psum += p;
        temp_probs[i] = p;
    }

    float ipsum = 1.0f / psum;
    for (int i = 0; i < num_candidates; ++i)
        temp_probs[i] *= ipsum;

//    printf("---- post\n");
//    DBGF(temperature);
//    printf("----\n");
//    for (int i = 0; i < num_candidates; ++i)
//        DBGIF(i, temp_probs[i]);
//    printf("\n");

    return num_candidates;
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
    const float* probs,
    const bool* logits_filter
)
{
    int maxidx = -1;
    float max = -1e38;

    for(int i = 1; i < num_candidates; i++)
    {
        if (logits_filter[i] && (maxidx == -1 || probs[i] > max))
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

inline bool cmp_asc(const float& a, const float& b)
{
    return a > b;
}

inline bool cmp_desc(const float& a, const float& b)
{
    return a < b;
}

template <bool (*cmp_func)(const float&, const float&)>
void quicksort_with_idx
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

        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        return;
    }

    if (high - low == 2)
    {
        int i0 = low;
        int i1 = low + 1;
        int i2 = low + 2;

        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (cmp_func(arr[i1], arr[i2])) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        return;
    }

    if (high - low == 3)
    {
        int i0 = low;
        int i1 = low + 1;
        int i2 = low + 2;
        int i3 = low + 3;

        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (cmp_func(arr[i1], arr[i2])) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (cmp_func(arr[i2], arr[i3])) { swap<float>(arr[i2], arr[i3]); swap<int>(idx[i2], idx[i3]); }
        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (cmp_func(arr[i1], arr[i2])) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        return;
    }

    if (high - low == 4)
    {
        int i0 = low;
        int i1 = low + 1;
        int i2 = low + 2;
        int i3 = low + 3;
        int i4 = low + 4;

        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (cmp_func(arr[i1], arr[i2])) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (cmp_func(arr[i2], arr[i3])) { swap<float>(arr[i2], arr[i3]); swap<int>(idx[i2], idx[i3]); }
        if (cmp_func(arr[i3], arr[i4])) { swap<float>(arr[i3], arr[i4]); swap<int>(idx[i3], idx[i4]); }
        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (cmp_func(arr[i1], arr[i2])) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (cmp_func(arr[i2], arr[i3])) { swap<float>(arr[i2], arr[i3]); swap<int>(idx[i2], idx[i3]); }
        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (cmp_func(arr[i1], arr[i2])) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        return;
    }

    float pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++)
    {
        if (!cmp_func(arr[j], pivot))
        {
            i++;
            swap<float>(arr[i], arr[j]);
            swap<int>(idx[i], idx[j]);
        }
    }

    swap<float>(arr[i + 1], arr[high]);
    swap<int>(idx[i + 1], idx[high]);
    int pos = i + 1;

    if (max_index == 0 || low <= max_index)
        quicksort_with_idx<cmp_func>(arr, idx, low, pos - 1, max_index);
    if (max_index == 0 || pos <= max_index)
        quicksort_with_idx<cmp_func>(arr, idx, pos + 1, high, max_index);
}

// Discard tiny probabilities, improves performance when temperature is very low

int pre_sort_descending
(
    const int num_candidates,
    float* arr,
    int* idx
)
{
    const float eps = 1e-8;
    int i = 0;
    int j = num_candidates - 1;

    while (i <= j)
    {
        if (arr[j] < eps) { j--; continue; }
        if (arr[i] >= eps) { i++; continue; }
        swap<float>(arr[i], arr[j]);
        swap<int>(idx[i], idx[j]);
        i++;
        j--;
    }

    return i;
}

int sort_descending
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    int max_index
)
{
    int pre = pre_sort_descending(num_candidates, temp_probs, temp_indices);
    quicksort_with_idx<cmp_desc>(temp_probs, temp_indices, 0, pre - 1, max_index);

//    int m = (max_index == 0 ? num_candidates : max_index);
//    for (int i = 0; i < m; i++) printf("%i - %f \n", temp_indices[i], temp_probs[i] * 10000.0);
//    for (int i = 0; i < m - 1; i++) if (temp_probs[i] < temp_probs[i + 1] - 2e-8) DBGI(i);

    return pre;
}

int top_k_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    int top_k
)
{
    //TIME_START;

    // Use min-heap for lower values of K

    if (top_k <= top_k_heap_threshold)
    {
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> min_heap;

        for (int i = 0; i < top_k; ++i) min_heap.push({temp_probs[i], temp_indices[i]});

        for (int i = top_k; i < num_candidates; i++)
        {
            if (temp_probs[i] > min_heap.top().first)
            {
                min_heap.pop();
                min_heap.push({temp_probs[i], temp_indices[i]});
            }
        }

        int j = top_k;
        for (int i = 0; i < top_k; i++)
        {
            j--;
            temp_probs[j] = min_heap.top().first;
            temp_indices[j] = min_heap.top().second;
            min_heap.pop();
        }
    }

    // For larger values, quicksort is still faster

    else
    {
        sort_descending(num_candidates, temp_probs, temp_indices, top_k);
    }

    //TIME_STOP;

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
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> min_heap;

    //TIME_START;

    float min_p = 1e-6;

    float sum = 0.0f;
    for (int i = 0; i < num_candidates; i++)
    {
        if (temp_probs[i] < min_p) continue;
        if (sum > top_p && temp_probs[i] < min_heap.top().first) continue;

        min_heap.push({temp_probs[i], temp_indices[i]});
        sum += temp_probs[i];

        while (sum > top_p && min_heap.size() > 1)
        {
            sum -= min_heap.top().first;
            min_heap.pop();
        }
    }

    int j = min_heap.size();
    int k = j;
    while (j > 0)
    {
        j--;
        temp_probs[j] = min_heap.top().first;
        temp_indices[j] = min_heap.top().second;
        min_heap.pop();
    }

    //TIME_STOP;

    return k;
}

int keep_threshold
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float threshold
)
{
    int i = 0;
    int j = num_candidates - 1;

    while (j >= i)
    {
        while (temp_probs[i] >= threshold && j >= i) i++;
        if (temp_probs[j] >= threshold)
        {
            swap<float>(temp_probs[i], temp_probs[j]);
            swap<int>(temp_indices[i], temp_indices[j]);
            i++;
        }
        j--;
    }
    return i;
}

int min_p_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float min_p
)
{
    //TIME_START;

    float top_prob = temp_probs[0];
    for (int i = 1; i < num_candidates; i++)
        if (temp_probs[i] > top_prob) top_prob = temp_probs[i];

    float threshold = top_prob * min_p;
    int n = keep_threshold(num_candidates, temp_probs, temp_indices, threshold);

    //TIME_STOP;

    return n;
}

int tfs_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float tfs
)
{
    //TIME_START;

    if (num_candidates < 3) return num_candidates;  // Discrete 2nd derivative undefined

    // 2nd derivative of sorted probs

    int nc = sort_descending(num_candidates, temp_probs, temp_indices, num_candidates);

    float* derivative = (float*) malloc(nc * sizeof(float));
    float dsum = 0.0f;
    for (int i = 0; i < nc - 2; i++)
    {
        float d = fabs(- temp_probs[i] + 2 * temp_probs[i + 1] - temp_probs[i + 2]);
        dsum += d;
        derivative[i] = d;
    }

    // Keep probs for cumulative sum of normalized 2nd derivative <= threshold

    float dsum_i = 1.0f / dsum;
    int k = 0;
    float cumsum = 0.0f;
    while (k < nc - 2)
    {
        cumsum += derivative[k] * dsum_i;
        if (cumsum > tfs) break;
        k++;
    }

    // Center distribution on the cutoff point

    k++;

    //TIME_STOP;

    free(derivative);
    return k;
}

int mirostat_pre_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float mirostat_mu,
    float mirostat_tau,
    float mirostat_eta
)
{
    //TIME_START;

    // If mu not yet initialized, initialize here

    float mu = mirostat_mu;
    if (mu == 0.0f) mu = mirostat_tau * 2.0f;

    // Discard tokens with surprise greater than mu

    int nc = sort_descending(num_candidates, temp_probs, temp_indices, num_candidates);

    float target_prob = powf(2, -mu);
    int k = 1;
    for (; k < nc; k++)
        if (temp_probs[k] < target_prob) break;

    //TIME_STOP;

    return k;
}

float mirostat_post_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float mirostat_mu,
    float mirostat_tau,
    float mirostat_eta
)
{
    // If mu not yet initializer, initialize here

    float mu = mirostat_mu;
    if (mu == 0.0f) mu = mirostat_tau * 2.0f;

    // Adjust mu based on probability of final choice

    float observed_surprise = -log2(temp_probs[0]);
    mu += mirostat_eta * (mirostat_tau - observed_surprise);

    return mu;
}

int typical_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float typical
)
{
    //TIME_START;

    const float epsilon = 1e-10;

    float* temp = (float*) malloc(num_candidates * sizeof(float));
    int* entropy_dev_order = (int*) malloc(num_candidates * sizeof(int));
    int* temp_indices_2 = (int*) malloc(num_candidates * sizeof(int));

    float neg_entropy = 0.0f;
    for (int i = 0; i < num_candidates; i++)
    {
        float x = temp_probs[i];
        float y = x + logf(x + epsilon);
        neg_entropy += x * y;
        temp[i] = y;  // temp = log_probs
    }

    for (int i = 0; i < num_candidates; i++)
    {
        temp[i] = fabs(temp[i] - neg_entropy);  // temp = entropy_dev
        entropy_dev_order[i] = i;
    }

    quicksort_with_idx<cmp_asc>(temp, entropy_dev_order, 0, num_candidates - 1, num_candidates);

    memcpy(temp, temp_probs, num_candidates * sizeof(float));  // temp = temp_probs
    memcpy(temp_indices_2, temp_indices, num_candidates * sizeof(int));

    float cumprob = 0.0f;
    int num = 0;

    while (true)
    {
        int j = entropy_dev_order[num];
        float p = temp[j];
        temp_probs[num] = p;
        temp_indices[num] = temp_indices_2[j];

        cumprob += p;
        if (cumprob >= typical) break;
        num++;
        if (num >= num_candidates) break;
    }

    free(temp);
    free(entropy_dev_order);
    free(temp_indices_2);

    //TIME_STOP;

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



