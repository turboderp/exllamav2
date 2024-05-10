
#include "generator.h"
#include "util.h"

// Compare string Q against list of strings S, utf-32 encoded and packed in byte array.
//
// Returns:
// -1: No matches
// -2: Partial match; at least one string in S partially overlaps Q on the right-hand side
// >= 0: Index into Q of full match with at least one string in S

int partial_strings_match
(
    py::buffer match,
    py::buffer offsets,
    py::buffer strings
)
{
    py::buffer_info info;

    info = match.request();
    uint32_t* q = static_cast<uint32_t*>(info.ptr);
    int q_len = info.size / 4;

    info = offsets.request();
    uint32_t* offsets_int = static_cast<uint32_t*>(info.ptr);
    int num_strings = info.size / 4 - 1;

    info = strings.request();
    uint32_t* strings_utf32 = static_cast<uint32_t*>(info.ptr);

    for (int i = 0; i < num_strings; ++i)
    {
        int beg = offsets_int[i] / 4;
        int s_len = offsets_int[i + 1] / 4 - beg;
        uint32_t* s = strings_utf32 + beg;

        int a = 0;
        int b = 0;
        while (a < q_len)
        {
            int a0 = a;
            while (q[a++] == s[b++])
            {
                if (b == s_len) return a0;
                if (a == q_len) return -2;
            }
            a = a0 + 1;
            b = 0;
       }
    }

    return -1;
}



