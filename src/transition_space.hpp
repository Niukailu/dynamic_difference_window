#ifndef __TRANSITION_SPACE_HPP__
#define __TRANSITION_SPACE_HPP__

#include <stdint.h>
#include "probability.hpp"

class transition_space {
public:
    int num, out_num; //A的个数，被删的在窗口外的A的个数
    uint32_t base;
    uint32_t A[32];
    probability proba;

    transition_space() { for(int i = 0; i < 32; i++) A[i] = 0; }
    ~transition_space() {}
    uint32_t size() { return 1U<<num; }
};

#endif