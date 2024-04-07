#ifndef _profiling_h
#define _profiling_h

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <string>
#include <map>

struct ProfileItem
{
    uint64_t min;
    uint64_t max;
    uint64_t total;
    uint64_t count;
};

void profile_start(std::string stage);
void profile_stop();
void profile_results();

#endif