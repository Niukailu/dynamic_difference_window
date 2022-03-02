#ifndef __HYPERPARAMETER_HPP__
#define __HYPERPARAMETER_HPP__
#include <cstring>


/**
 * FROM_0x0_0x1_TO_0x1_0x0_ROUND_39
 * FROM_0x0_0x11_TO_0x5_0x2_ROUND_28
 * FROM_0x440_0x1880_TO_0x440_0x100_ROUND_22
 * FROM_0x4000000_0x11000000_TO_0x11000000_0x4000000_ROUND_21
**/
#define FROM_0x4000000_0x11000000_TO_0x11000000_0x4000000_ROUND_21


#ifdef FROM_0x0_0x1_TO_0x1_0x0_ROUND_39
    #define PRECISION 14        // 窗口大小
    #define ROUNDS 40           // 轮数
    #define LOAD_ROUND 0        // 程序断掉之后加载的轮数，0不加载
    const std::string name = "FROM_0x0_0x1_TO_0x1_0x0";    // 程序保存的名字，将中间结果保存在 experiments/{name}/{round}/* 内
    const uint32_t begin_left = 0;
    const uint32_t begin_right = 1;
    const uint32_t end_left = 1;
    const uint32_t end_right = 0;
#endif

#ifdef FROM_0x0_0x11_TO_0x5_0x2_ROUND_27
    #define PRECISION 14
    #define ROUNDS 28
    #define LOAD_ROUND 0
    const std::string name = "FROM_0x0-0x11_TO_0x5_0x2";
    const uint32_t begin_left = 0;
    const uint32_t begin_right = 17;
    const uint32_t end_left = 5;
    const uint32_t end_right = 2;
#endif

#ifdef FROM_0x440_0x1880_TO_0x440_0x100_ROUND_22
    #define PRECISION 14
    #define ROUNDS 23
    #define LOAD_ROUND 0
    const std::string name = "FROM_0x440_0x1880_TO_0x440_0x100";
    const uint32_t begin_left = 1088;
    const uint32_t begin_right = 6272;
    const uint32_t end_left = 1088;
    const uint32_t end_right = 256;
#endif

#ifdef FROM_0x4000000_0x11000000_TO_0x11000000_0x4000000_ROUND_21
    #define PRECISION 14
    #define ROUNDS 22
    #define LOAD_ROUND 0
    const std::string name = "FROM_0x4000000_0x11000000_TO_0x11000000_0x4000000_ROUND_21";
    const uint32_t begin_left = 67108864;
    const uint32_t begin_right = 285212672;
    const uint32_t end_left = 285212672;
    const uint32_t end_right = 67108864;
#endif

#endif
