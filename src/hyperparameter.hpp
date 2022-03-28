#ifndef __HYPERPARAMETER_HPP__
#define __HYPERPARAMETER_HPP__
#include <cstring>


/**
 * SIMECK64_FROM_0x0_0x1_TO_0x1_0x0_ROUND_39
 * SIMECK64_FROM_0x0_0x11_TO_0x5_0x2_ROUND_27
 * SIMON64_FROM_0x440_0x1880_TO_0x440_0x100_ROUND_22
 * SIMON64_FROM_0x4000000_0x11000000_TO_0x11000000_0x4000000_ROUND_21
 * SIMON48_FROM_0x80_0x222_TO_0x222_0x80_ROUND_17
 * SIMON32_FROM_0x0_0x40_TO_0x4000_0x0_ROUND_13
**/
// 在这里定义启用哪一个实验
#define SIMON64_FROM_0x440_0x1880_TO_0x440_0x100_ROUND_22


#ifdef SIMECK64_FROM_0x0_0x1_TO_0x1_0x0_ROUND_39
    #define SIMECK64            // 使用SIMECK64
    #define PRECISION 14        // 窗口大小
    #define ROUNDS 40           // 轮数
    #define LOAD_ROUND 0        // 程序断掉之后加载的轮数，0不加载
    const std::string name = "SIMECK64_FROM_0x0_0x1_TO_0x1_0x0_ROUND_39";    // 程序保存的名字，将中间结果保存在 experiments/{name}/{round}/* 内
    const uint32_t begin_left = 0;
    const uint32_t begin_right = 1;
    const uint32_t end_left = 1;
    const uint32_t end_right = 0;
#endif

#ifdef SIMECK64_FROM_0x0_0x11_TO_0x5_0x2_ROUND_27
    #define SIMECK64
    #define PRECISION 14
    #define ROUNDS 28
    #define LOAD_ROUND 0
    const std::string name = "SIMECK64_FROM_0x0_0x11_TO_0x5_0x2_ROUND_27";
    const uint32_t begin_left = 0;
    const uint32_t begin_right = 17;
    const uint32_t end_left = 5;
    const uint32_t end_right = 2;
#endif

#ifdef SIMON64_FROM_0x440_0x1880_TO_0x440_0x100_ROUND_22
    #define SIMON64
    #define PRECISION 14
    #define ROUNDS 23
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_FROM_0x440_0x1880_TO_0x440_0x100_ROUND_22";
    const uint32_t begin_left = 1088;
    const uint32_t begin_right = 6272;
    const uint32_t end_left = 1088;
    const uint32_t end_right = 256;
#endif

#ifdef SIMON64_FROM_0x4000000_0x11000000_TO_0x11000000_0x4000000_ROUND_21
    #define SIMON64
    #define PRECISION 14
    #define ROUNDS 22
    #define LOAD_ROUND 0
    const std::string name = "SIMON64_FROM_0x4000000_0x11000000_TO_0x11000000_0x4000000_ROUND_21";
    const uint32_t begin_left = 67108864;
    const uint32_t begin_right = 285212672;
    const uint32_t end_left = 285212672;
    const uint32_t end_right = 67108864;
#endif

#ifdef SIMON48_FROM_0x80_0x222_TO_0x222_0x80_ROUND_17
    #define SIMON48
    #define PRECISION 14
    #define ROUNDS 18
    #define LOAD_ROUND 0
    const std::string name = "SIMON48_FROM_0x80_0x222_TO_0x222_0x80_ROUND_17";
    const uint32_t begin_left = 128;
    const uint32_t begin_right = 546;
    const uint32_t end_left = 546;
    const uint32_t end_right = 128;
#endif

#ifdef SIMON32_FROM_0x0_0x40_TO_0x4000_0x0_ROUND_13
    #define SIMON32
    #define PRECISION 14
    #define ROUNDS 14
    #define LOAD_ROUND 0
    const std::string name = "SIMON32_FROM_0x0_0x40_TO_0x4000_0x0_ROUND_13";
    const uint16_t begin_left = 0;
    const uint16_t begin_right = 64;
    const uint16_t end_left = 16384;
    const uint16_t end_right = 0;
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
