#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include "src/simeck.hpp"
using namespace std;

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
mt19937 rand_num(seed);


void gen_input(uint32_t keys[], uint32_t in[]) {
    for(int i = 0; i < 4; i++) keys[i] = rand_num();
    for(int i = 0; i < 2; i++) in[i] = rand_num();
}


int main() {
    int num = 0;
    int in_left = 0, in_right = 1, out_left=1, out_right=0, round = 11;
    uint32_t keys[4], in1[2], in2[2], out1[2], out2[2];

    for(long long epoch = 1; ; epoch++) {
        gen_input(keys, in1);
        in2[1] = in1[1] ^ in_left;
        in2[0] = in1[0] ^ in_right;

        simeck_64_128(keys, in1, out1, round);
        simeck_64_128(keys, in2, out2, round);
        if(out_left == (out1[1]^out2[1]) && out_right == (out1[0]^out2[0])) {
            num++;
        }
        if(epoch % 100000 == 0) cout << "[" << epoch << "] prob is: " << log2(1.0 * num / epoch) << endl; 
    }
    return 0;
}
