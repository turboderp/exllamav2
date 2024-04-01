#include "profiling.h"
#include "../config.h"

// Profiling functions

std::map<std::string, ProfileItem> all_stages = {};
std::string current_stage = "";
std::chrono::time_point<std::chrono::high_resolution_clock> stage_start;

void profile_start(std::string stage)
{
#ifdef SAMPLE_PROFILING
    current_stage = stage;
    stage_start = std::chrono::high_resolution_clock::now();
#endif
}

void profile_stop()
{
#ifdef SAMPLE_PROFILING
    auto stage_stop = std::chrono::high_resolution_clock::now();
    uint64_t duration = std::chrono::duration_cast<std::chrono::microseconds>(stage_stop - stage_start).count();
    auto r = all_stages.find(current_stage);
    if (r == all_stages.end())
    {
        ProfileItem item;
        item.min = duration;
        item.max = duration;
        item.total = duration;
        item.count = 1;
        all_stages[current_stage] = item;
    }
    else
    {
        ProfileItem* item = &(r->second);
        item->total += duration;
        item->min = std::min(item->min, duration);
        item->max = std::max(item->max, duration);
        item->count++;
    }
#endif
}

void profile_results()
{
#ifdef SAMPLE_PROFILING
    printf("stage                               total             min             max            mean\n");
    printf("-----------------------------------------------------------------------------------------\n");
    for (const auto& entry : all_stages)
    {
        printf("%26s %11llu us  %11llu us  %11llu us  %11llu us\n", entry.first.c_str(), entry.second.total, entry.second.min, entry.second.max, entry.second.total / entry.second.count);
    }
#endif
}
