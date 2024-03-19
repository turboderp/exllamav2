
#ifndef _q_gemm_autotune_cuh
#define _q_gemm_autotune_cuh

#define AT_SHAPEHASH(device, size_m, size_k, size_n) (((uint64_t)device << 48) | ((uint64_t)size_m << 32) | ((uint64_t)size_n << 16) | ((uint64_t)size_k))
#define AT_NUM_MEASURE 160

struct AT_Result
{
    std::vector<float> timings_32;
    std::vector<float> timings_64;
    int best;
};

std::map<uint64_t, AT_Result> at_results = {};

AT_Result* at_get(int device, int size_m, int size_k, int size_n)
{
    uint64_t hash = AT_SHAPEHASH(device, size_m, size_k, size_n);
    if (at_results.find(hash) == at_results.end())
    {
        AT_Result r;
        r.best = 0;
        at_results[hash] = r;
    }
    auto it = at_results.find(hash);
    return &(it->second);
}

AT_Result* at_get_top(int device, int size_k, int size_n)
{
    for (int size_m = EXL2_BLOCK_M_SIZE_MAX; size_m >= 1; --size_m)
    {
        uint64_t hash = AT_SHAPEHASH(device, size_m, size_k, size_n);
        auto it = at_results.find(hash);
        if (it != at_results.end()) return &(it->second);
    }
    return NULL;
}

float iqm(std::vector<float>& v)
{
    std::sort(v.begin(), v.end());
    int p0 = v.size() * 1 / 4;
    int p1 = v.size() * 3 / 4;
    float sum = 0.0f;
    for (int i = p0; i < p1; ++i) sum += v[i];
    return sum / (float)(p1 - p0);
}

void at_select(AT_Result* atr)
{
    float iqm_32 = iqm(atr->timings_32);
    float iqm_64 = iqm(atr->timings_64);
    atr->best = iqm_32 < iqm_64 ? 32 : 64;
    // DBGF3((float) r->best, iqm_32, iqm_64);
    atr->timings_32.clear();
    atr->timings_64.clear();
}

int at_get_fallback_blocksize(int device, int size_m, int size_k, int size_n)
{
    return max(size_k, size_n) < 12000 ? 32 : 64;
}

#endif