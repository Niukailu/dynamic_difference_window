#ifndef __HYPERPARAMETER_HPP__
#define __HYPERPARAMETER_HPP__
#include <cstring>


/**
 * SIMECK64_FROM_0x0_0x1_PRECISION_14
 * SIMECK64_FROM_0x0_0x11_PRECISION_14
 * SIMON32_FROM_0x0_0x40_PRECISION_14
 * SIMON48_FROM_0x80_0x222_PRECISION_14
 * SIMON64_FROM_0x440_0x1880_PRECISION_14
 * SIMON64_FROM_0x4000000_0x11000000_PRECISION_14
**/
// 在这里定义启用哪一个实验
#define SIMON64_FROM_0x440_0x1880_PRECISION_14

#ifdef SIMECK64_FROM_0x0_0x1_PRECISION_14
    #define SIMECK64            // 使用SIMECK64
    #define PRECISION 14        // 窗口大小
    #define LOAD_ROUND 0        // 程序断掉之后加载的轮数，0不加载
    const std::string name = "SIMECK64_FROM_0x0_0x1_PRECISION_14";    // 程序保存的名字，将中间结果保存在 experiments/{name}/{round}/* 内
    const uint32_t begin_left = 0x0;
    const uint32_t begin_right = 0x1;
#endif

#ifdef SIMECK64_FROM_0x0_0x11_PRECISION_14
    #define SIMECK64
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMECK64_FROM_0x0_0x11_PRECISION_14";
    const uint32_t begin_left = 0x0;
    const uint32_t begin_right = 0x11;
#endif

#ifdef SIMON32_FROM_0x0_0x40_PRECISION_14
    #define SIMON32
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON32_FROM_0x0_0x40_PRECISION_14";
    const uint16_t begin_left = 0x0;
    const uint16_t begin_right = 0x40;
#endif

#ifdef SIMON48_FROM_0x80_0x222_PRECISION_14
    #define SIMON48
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON48_FROM_0x80_0x222_PRECISION_14";
    const uint32_t begin_left = 0x80;
    const uint32_t begin_right = 0x222;
#endif

#ifdef SIMON64_FROM_0x440_0x1880_PRECISION_14
    #define SIMON64
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_FROM_0x440_0x1880_PRECISION_14";
    const uint32_t begin_left = 0x440;
    const uint32_t begin_right = 0x1880;
#endif

#ifdef SIMON64_FROM_0x4000000_0x11000000_PRECISION_14
    #define SIMON64
    #define PRECISION 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_FROM_0x4000000_0x11000000_PRECISION_14";
    const uint32_t begin_left = 0x4000000;
    const uint32_t begin_right = 0x11000000;
#endif

#if defined(SIMECK32)+defined(SIMECK48)+defined(SIMECK64)+defined(SIMECK96)+defined(SIMECK128)+defined(SIMON32)+defined(SIMON48)+defined(SIMON64)+defined(SIMON96)+defined(SIMON128) != 1
    #error Please only exactly one of SIMON and SIMECK
#endif

#if defined(SIMECK32) | defined(SIMON32)
    const int BITS = 16;
    typedef uint16_t dtype;
    #define ROT(x, r) (((x) << (r)) | ((x) >> (16 - (r))))
#endif

#if defined(SIMECK48) | defined(SIMON48)
    const int BITS = 24;
    typedef uint32_t dtype;
    #define ROT(x, r) ((((x) << (r)) % (1 << 24)) | ((x) >> (24 - (r))))
#endif

#if defined(SIMECK64) | defined(SIMON64)
    const int BITS = 32;
    typedef uint32_t dtype;
    #define ROT(x, r) (((x) << (r)) | ((x) >> (32 - (r))))
#endif

#if defined(SIMECK96) | defined(SIMON96)
    const int BITS = 48;
    typedef uint64_t dtype;
    #define ROT(x, r) ((((x) << (r)) % (1 << 48)) | ((x) >> (48 - (r))))
#endif

#if defined(SIMECK128) | defined(SIMON128)
    const int BITS = 64;
    typedef uint64_t dtype;
    #define ROT(x, r) (((x) << (r)) | ((x) >> (64 - (r))))
#endif

#if defined(SIMECK32) | defined(SIMECK48) | defined(SIMECK64) | defined(SIMECK96) | defined(SIMECK128)
    inline dtype f(dtype x) { return (ROT(x, 5) & x) ^ ROT(x, 1); }
#endif

#if defined(SIMON32) | defined(SIMON48) | defined(SIMON64) | defined(SIMON96) | defined(SIMON128)
    inline dtype f(dtype x) { return (ROT(x, 8) & ROT(x, 1)) ^ ROT(x, 2); }
#endif

#endif
