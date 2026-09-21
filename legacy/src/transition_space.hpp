#ifndef __TRANSITION_SPACE_HPP__
#define __TRANSITION_SPACE_HPP__

#include <stdint.h>
#include "probability.hpp"
#include "hyperparameter.hpp"

class transition_space {
public:
    int num, out_num; //A的个数，被删的在窗口外的A的个数
    dtype base;
    dtype A[BITS];
    probability proba;

    transition_space() { for(int i = 0; i < BITS; i++) A[i] = 0; }
    ~transition_space() {}
};

#endif