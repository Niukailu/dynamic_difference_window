#ifndef __HYPERPARAMETER_HPP__
#define __HYPERPARAMETER_HPP__
#include <cstring>


/**
 * SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_14
 * SIMON64_DIFFERENCE_FROM_0x1_0x40000004_PRECISION_14
 * SIMON64_DIFFERENCE_FROM_0x4000000_0x11000000_PRECISION_14
 * SIMON64_DIFFERENCE_FROM_0x440_0x1880_PRECISION_14
 * SIMON64_DIFFERENCE_FROM_0x80000_0x222000_PRECISION_14
 * SIMON64_LINEAR_FROM_0x44400_0x1000_PRECISION_14
 * SIMON64_LINEAR_FROM_0x40000004_0x1_PRECISION_14
 * SIMON96_DIFFERENCE_FROM_0x100000_0x444040_PRECISION_14
 * SIMON96_DIFFERENCE_FROM_0x4000_0x11101_PRECISION_14
 * SIMON96_LINEAR_FROM_0x1_0x0_PRECISION_14
 * SIMON96_LINEAR_FROM_0x400000004044_0x1_PRECISION_14
 * SIMON96_LINEAR_FROM_0x400000000044_0x1_PRECISION_14
 * SIMON128_DIFFERENCE_FROM_0x0_0x1_PRECISION_14
 * SIMON128_DIFFERENCE_FROM_0x1000_0x4440_PRECISION_14
 * SIMON128_LINEAR_FROM_0x4000000000000004_0x1_PRECISION_14
 * SIMON128_LINEAR_FROM_0x4000000000000044_0x1_PRECISION_14
 * SIMON128_LINEAR_FROM_0x1_0x0_PRECISION_14
 * SIMON128_LINEAR_FROM_0x200000_0x880000_PRECISION_14
**/
// 在这里定义启用哪一个实验
#define SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_13

/*************************** SIMON64 ***************************/
#ifdef SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_7
    #define SIMON64
    #define DIFFERENCE
    #define PRECISION 7
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_7";
    const uint32_t begin_left = 0x0;
    const uint32_t begin_right = 0x1;
#endif

#ifdef SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_8
    #define SIMON64
    #define DIFFERENCE
    #define PRECISION 8
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_8";
    const uint32_t begin_left = 0x0;
    const uint32_t begin_right = 0x1;
#endif

#ifdef SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_9
    #define SIMON64
    #define DIFFERENCE
    #define PRECISION 9
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_9";
    const uint32_t begin_left = 0x0;
    const uint32_t begin_right = 0x1;
#endif

#ifdef SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_10
    #define SIMON64
    #define DIFFERENCE
    #define PRECISION 10
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_10";
    const uint32_t begin_left = 0x0;
    const uint32_t begin_right = 0x1;
#endif

#ifdef SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_11
    #define SIMON64
    #define DIFFERENCE
    #define PRECISION 11
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_11";
    const uint32_t begin_left = 0x0;
    const uint32_t begin_right = 0x1;
#endif

#ifdef SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_12
    #define SIMON64
    #define DIFFERENCE
    #define PRECISION 12
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_12";
    const uint32_t begin_left = 0x0;
    const uint32_t begin_right = 0x1;
#endif

#ifdef SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_13
    #define SIMON64
    #define DIFFERENCE
    #define PRECISION 13
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_13";
    const uint32_t begin_left = 0x0;
    const uint32_t begin_right = 0x1;
#endif

#ifdef SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_14
    #define SIMON64
    #define DIFFERENCE
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_DIFFERENCE_FROM_0x0_0x1_PRECISION_14";
    const uint32_t begin_left = 0x0;
    const uint32_t begin_right = 0x1;
#endif

#ifdef SIMON64_DIFFERENCE_FROM_0x1_0x40000004_PRECISION_14
    #define SIMON64
    #define DIFFERENCE
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_DIFFERENCE_FROM_0x1_0x40000004_PRECISION_14";
    const uint32_t begin_left = 0x1;
    const uint32_t begin_right = 0x40000004;
#endif

#ifdef SIMON64_DIFFERENCE_FROM_0x4000000_0x11000000_PRECISION_14
    #define SIMON64
    #define DIFFERENCE
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_DIFFERENCE_FROM_0x4000000_0x11000000_PRECISION_14";
    const uint32_t begin_left = 0x4000000;
    const uint32_t begin_right = 0x11000000;
#endif

#ifdef SIMON64_DIFFERENCE_FROM_0x440_0x1880_PRECISION_14
    #define SIMON64
    #define DIFFERENCE
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_DIFFERENCE_FROM_0x440_0x1880_PRECISION_14";
    const uint32_t begin_left = 0x440;
    const uint32_t begin_right = 0x1880;
#endif

#ifdef SIMON64_DIFFERENCE_FROM_0x80000_0x222000_PRECISION_14
    #define SIMON64
    #define DIFFERENCE
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_DIFFERENCE_FROM_0x80000_0x222000_PRECISION_14";
    const uint32_t begin_left = 0x80000;
    const uint32_t begin_right = 0x222000;
#endif

#ifdef SIMON64_LINEAR_FROM_0x44400_0x1000_PRECISION_14
    #define SIMON64
    #define LINEAR
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_LINEAR_FROM_0x44400_0x1000_PRECISION_14";
    const uint64_t begin_left = 0x1000;
    const uint64_t begin_right = 0x44400;
#endif

#ifdef SIMON64_LINEAR_FROM_0x40000004_0x1_PRECISION_14
    #define SIMON64
    #define LINEAR
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_LINEAR_FROM_0x40000004_0x1_PRECISION_14";
    const uint64_t begin_left = 0x1;
    const uint64_t begin_right = 0x40000004;
#endif

/*************************** SIMON96 ***************************/
#ifdef SIMON96_DIFFERENCE_FROM_0x100000_0x444040_PRECISION_14
    #define SIMON96
    #define DIFFERENCE
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON96_DIFFERENCE_FROM_0x100000_0x444040_PRECISION_14";
    const uint64_t begin_left = 0x100000;
    const uint64_t begin_right = 0x444040;
#endif

#ifdef SIMON96_DIFFERENCE_FROM_0x4000_0x11101_PRECISION_14
    #define SIMON96
    #define DIFFERENCE
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON96_DIFFERENCE_FROM_0x4000_0x11101_PRECISION_14";
    const uint64_t begin_left = 0x4000;
    const uint64_t begin_right = 0x11101;
#endif

#ifdef SIMON96_LINEAR_FROM_0x1_0x0_PRECISION_14
    #define SIMON96
    #define LINEAR
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON96_LINEAR_FROM_0x1_0x0_PRECISION_14";
    const uint64_t begin_left = 0x0;
    const uint64_t begin_right = 0x1;
#endif


#ifdef SIMON96_LINEAR_FROM_0x400000004044_0x1_PRECISION_14
    #define SIMON96
    #define LINEAR
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON96_LINEAR_FROM_0x400000004044_0x1_PRECISION_14";
    const uint64_t begin_left = 0x1;
    const uint64_t begin_right = 0x400000004044;
#endif


#ifdef SIMON96_LINEAR_FROM_0x400000000044_0x1_PRECISION_14
    #define SIMON96
    #define LINEAR
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON96_LINEAR_FROM_0x400000000044_0x1_PRECISION_14";
    const uint64_t begin_left = 0x1;
    const uint64_t begin_right = 0x400000000044;
#endif

/*************************** SIMON128 ***************************/
#ifdef SIMON128_DIFFERENCE_FROM_0x0_0x1_PRECISION_14
    #define SIMON128
    #define DIFFERENCE
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON128_DIFFERENCE_FROM_0x0_0x1_PRECISION_14";
    const uint64_t begin_left = 0x0;
    const uint64_t begin_right = 0x1;
#endif

#ifdef SIMON128_DIFFERENCE_FROM_0x1000_0x4440_PRECISION_14
    #define SIMON128
    #define DIFFERENCE
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON128_DIFFERENCE_FROM_0x1000_0x4440_PRECISION_14";
    const uint64_t begin_left = 0x1000;
    const uint64_t begin_right = 0x4440;
#endif

#ifdef SIMON128_LINEAR_FROM_0x4000000000000004_0x1_PRECISION_14
    #define SIMON128
    #define LINEAR
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON128_LINEAR_FROM_0x4000000000000004_0x1_PRECISION_14";
    const uint64_t begin_left = 0x1;
    const uint64_t begin_right = 0x4000000000000004;
#endif

#ifdef SIMON128_LINEAR_FROM_0x4000000000000044_0x1_PRECISION_14
    #define SIMON128
    #define LINEAR
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON128_LINEAR_FROM_0x4000000000000044_0x1_PRECISION_14";
    const uint64_t begin_left = 0x1;
    const uint64_t begin_right = 0x4000000000000044;
#endif

#ifdef SIMON128_LINEAR_FROM_0x1_0x0_PRECISION_14
    #define SIMON128
    #define LINEAR
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON128_LINEAR_FROM_0x1_0x0_PRECISION_14";
    const uint64_t begin_left = 0x0;
    const uint64_t begin_right = 0x1;
#endif

#ifdef SIMON128_LINEAR_FROM_0x200000_0x880000_PRECISION_14
    #define SIMON128
    #define LINEAR
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON128_LINEAR_FROM_0x200000_0x880000_PRECISION_14";
    const uint64_t begin_left = 0x880000;
    const uint64_t begin_right = 0x200000;
#endif

#if defined(SIMECK32)+defined(SIMECK48)+defined(SIMECK64)+defined(SIMECK96)+defined(SIMECK128)+defined(SIMON32)+defined(SIMON48)+defined(SIMON64)+defined(SIMON96)+defined(SIMON128) != 1
    #error Please only exactly one of SIMON and SIMECK
#endif

#if defined(SIMECK32)+defined(SIMECK48)+defined(SIMECK64)+defined(SIMECK96)+defined(SIMECK128) == 1
    #define SIMECK
#endif

#if defined(SIMON32)+defined(SIMON48)+defined(SIMON64)+defined(SIMON96)+defined(SIMON128) == 1
    #define SIMON
#endif

#if defined(DIFFERENCE)+defined(LINEAR) != 1
    #error Please only exactly one of SIMON and SIMECK
#endif

#if defined(SIMECK32) | defined(SIMON32)
    const int BITS = 16;
    typedef uint16_t dtype;
    #define _ROT(x, r) (((x) << (r)) | ((x) >> (16 - (r))))
#endif

#if defined(SIMECK48) | defined(SIMON48)
    const int BITS = 24;
    typedef uint32_t dtype;
    #define _ROT(x, r) ((((x) << (r)) % (1 << 24)) | ((x) >> (24 - (r))))
#endif

#if defined(SIMECK64) | defined(SIMON64)
    const int BITS = 32;
    typedef uint32_t dtype;
    #define _ROT(x, r) (((x) << (r)) | ((x) >> (32 - (r))))
#endif

#if defined(SIMECK96) | defined(SIMON96)
    const int BITS = 48;
    typedef uint64_t dtype;
    #define _ROT(x, r) ((((x) << (r)) % (1ULL << 48)) | ((x) >> (48 - (r))))
#endif

#if defined(SIMECK128) | defined(SIMON128)
    const int BITS = 64;
    typedef uint64_t dtype;
    #define _ROT(x, r) (((x) << (r)) | ((x) >> (64 - (r))))
#endif

#if defined(SIMECK32) | defined(SIMECK48) | defined(SIMECK64) | defined(SIMECK96) | defined(SIMECK128)
    inline dtype f(dtype x) { return (_ROT(x, 5) & x) ^ _ROT(x, 1); }
#endif

#if defined(SIMON32) | defined(SIMON48) | defined(SIMON64) | defined(SIMON96) | defined(SIMON128)
    inline dtype f(dtype x) { return (_ROT(x, 8) & _ROT(x, 1)) ^ _ROT(x, 2); }
#endif

// _ROT is fast, and ROT is complete
inline dtype ROT(dtype x, int r) {
    r = (BITS + r % BITS) % BITS;
    if(BITS == 24) return ((x << r) % (1 << 24)) | (x >> (BITS - r));      // 24
    if(BITS == 48) return ((x << r) % (1ULL << 48)) | (x >> (BITS - r));   // 48
    return (x << r) | (x >> (BITS - r));     // else
}

#endif
