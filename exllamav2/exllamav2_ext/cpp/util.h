#ifndef _util_h
#define _util_h

#include <chrono>

#define DBGS(__x) printf("%s\n", __x)
#define DBGI(__x) printf("%s: %li\n", #__x, __x)
#define DBGI2(__x, __y) printf("%s, %s: %li, %li\n", #__x, #__y, __x, __y)
#define DBGI3(__x, __y, __z) printf("%s, %s, %s: %li, %li, %li\n", #__x, #__y, #__z, __x, __y, __z)
#define DBGX(__x) printf("%s: %lx\n", #__x, __x)
#define DBGX2(__x, __y) printf("%s, %s: %lx, %lx\n", #__x, #__y, __x, __y)
#define DBGX3(__x, __y, __z) printf("%s, %s, %s: %lx, %lx, %lx\n", #__x, #__y, #__z, __x, __y, __z)

#define DBGF(__x) printf("%s: %f\n", #__x, __x)
#define DBGF2(__x, __y) printf("%s, %s: %f, %f\n", #__x, #__y, __x, __y)
#define DBGF3(__x, __y, __z) printf("%s, %s, %s: %f, %f, %f\n", #__x, #__y, #__z, __x, __y, __z)
#define DBGF4(__x, __y, __z, __w) printf("%s, %s, %s, %s: %f, %f, %f, %f\n", #__x, #__y, #__z, #__w, __x, __y, __z, __w)
#define DBGIF(__x, __y) printf("%s, %s: %i, %f\n", #__x, #__y, __x, __y)

#define TIME_START \
    auto start = std::chrono::high_resolution_clock::now()

#define TIME_STOP \
    do { \
        auto stop = std::chrono::high_resolution_clock::now(); \
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); \
        DBGI(duration_us); \
    } while (false)

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

// Some decluttering macros

#define TORCH_CHECK_DTYPE(__x, __dtype) TORCH_CHECK((__x).dtype() == torch::__dtype, #__x " is incorrect datatype, must be " #__dtype)
#define TORCH_CHECK_DTYPE_OPT(__x, __dtype) TORCH_CHECK((__x).device().is_meta() || (__x).dtype() == torch::__dtype, #__x " is incorrect datatype, must be " #__dtype)
#define TORCH_CHECK_SHAPES(__x, __dim_x, __y, __dim_y, __scale_y) TORCH_CHECK((__x).size(__dim_x) == (__y).size(__dim_y) * __scale_y, #__x " and " #__y " have incompatible shapes")
#define TORCH_CHECK_SHAPES_OPT(__x, __dim_x, __y, __dim_y, __scale_y) TORCH_CHECK((__x).device().is_meta() || (__x).size(__dim_x) == (__y).size(__dim_y) * __scale_y, #__x " and " #__y " have incompatible shapes")
#define TORCH_CHECK_SHAPES_FULL(__x, __y) TORCH_CHECK((__x).sizes() == (__y).sizes(), #__x " and " #__y " have incompatible shapes")
#define TORCH_CHECK_NUMEL(__x, __y) TORCH_CHECK((__x).numel() == (__y).numel(), #__x " and " #__y " have incompatible shapes")

#endif
