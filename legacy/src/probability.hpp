#ifndef __PROBABILITY__
#define __PROBABILITY__

#include <math.h>
#include <algorithm>
const double ZERO = log2(0);

class probability {  //概率矩阵中存的概率
public:
    double value;
    probability() { value = ZERO; }
    probability(double value) { this->value = log2(value); }
    ~probability() {} //析构函数
    probability operator + (probability& other) const {
        probability ret;
        double a = value, b = other.value;
        if(a < b) std::swap(a, b);
        ret.value = log2(1 + pow(2, b - a)) + a;
        return ret;
    }
    probability operator - (probability& other) const {
        probability ret;
        double a = value, b = other.value;
        if (a > b) ret.value = log2(1 - pow(2, b - a)) + a;
        else ret.value = log2(pow(2, a - b) - 1) + b;
        return ret;
    }
    probability operator += (probability other) {
        double a = value, b = other.value;
        if(a < b) std::swap(a, b);
        if(a == ZERO) return *this;
        this->value = log2(1 + pow(2, b - a)) + a;
        return *this;
    }
    probability operator-= (probability& other) {
        double a = value, b = other.value;
        if (a > b) this->value = log2(1 - pow(2, b - a)) + a;
        else this->value = log2(pow(2, a - b) - 1) + b;
        return *this;
    }
    probability operator * (const probability& other) const {
        probability ret;
        ret.value = value + other.value;
        return ret;
    }
    probability operator /= (const probability& other) {
        this->value = value - other.value;
        return *this;
    }
    bool operator < (probability other) const { return value < other.value; }
    bool operator > (probability other) const { return value > other.value; }
    bool operator > (double other) const { return value > log2(other); }
    bool operator == (probability other) const { return value == other.value; }
    double operator() () const { return pow(2, value); }
};

const probability TWO = probability(2);
const probability zo6 = probability(0.999999);

#endif