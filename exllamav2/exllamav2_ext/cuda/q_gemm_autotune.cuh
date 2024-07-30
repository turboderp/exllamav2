
#ifndef _q_gemm_autotune_cuh
#define _q_gemm_autotune_cuh

#if defined(USE_ROCM)
// Autotune seems unreliable on ROCm
#define AT_USE_GEMM_AUTOTUNE false
#else
#define AT_USE_GEMM_AUTOTUNE true
#endif
#define AT_SHAPEHASH(device, size_m, size_k, size_n) (((uint64_t)device << 48) | ((uint64_t)size_m << 32) | ((uint64_t)size_n << 16) | ((uint64_t)size_k))
#define AT_NUM_MEASURE 200
#define AT_NUM_WARMUP 20

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

std::tuple<float, float, float> iqm(std::vector<float>& v)
{
    std::sort(std::next(v.begin(), AT_NUM_WARMUP), v.end());
    int p0 = (v.size() - AT_NUM_WARMUP) * 5 / 16 + AT_NUM_WARMUP;
    int p1 = (v.size() - AT_NUM_WARMUP) * 11 / 16 + AT_NUM_WARMUP;
    float sum = 0.0f;
    for (int i = p0; i < p1; ++i) sum += v[i];
    return std::tuple<float, float, float> (sum / (float)(p1 - p0), v[p0], v[p1 - 1]);
}

int at_get_fallback_blocksize(int device, int size_m, int size_k, int size_n)
{
    #if defined(USE_ROCM)
        return 64;
    #else
        return max(size_k, size_n) < 12000 ? 32 : 64;
    #endif
}

void at_select(AT_Result* atr, int device, int size_m, int size_k, int size_n)
{
    std::tuple<float, float, float> r32 = iqm(atr->timings_32);
    std::tuple<float, float, float> r64 = iqm(atr->timings_64);
    float iqm32 = std::get<0>(r32);
    float min32 = std::get<1>(r32);
    float max32 = std::get<2>(r32);
    float iqm64 = std::get<0>(r64);
    float min64 = std::get<1>(r64);
    float max64 = std::get<2>(r64);
    float diff = abs(iqm32 - iqm64) / (iqm32 + 1e-30);
    if (max32 > min32 * 1.15 || diff < 0.025)
        atr->best = at_get_fallback_blocksize(device, size_m, size_k, size_n);
    else
        atr->best = iqm32 <= iqm64 ? 32 : 64;
//    DBGF2(min32, max32);
//    DBGF2(min64, max64);
//    DBGF3((float) atr->best, iqm32, iqm64);
    atr->timings_32.clear();
    atr->timings_64.clear();
}

#endif