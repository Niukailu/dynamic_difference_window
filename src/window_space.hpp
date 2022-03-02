#ifndef __WINDOW_SPACE_HPP__
#define __WINDOW_SPACE_HPP__

#include <stdint.h>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "hyperparameter.hpp"

class window_space {  //窗口位置坐标
public:
    uint32_t base;
    uint32_t mask;
    uint32_t inv_mask;
    std::vector<uint32_t> data;

    window_space() {}
    window_space(uint32_t base, std::vector<int> active_bits) {
        this->base = base;
        int active_bits_size = active_bits.size();
        assert(active_bits_size <= PRECISION);
        std::sort(active_bits.begin(), active_bits.end());
        for(int i = 1; i < active_bits_size; i++)
            assert(active_bits[i - 1] != active_bits[i]);
        mask = 0;
        for(int i = 0; i < active_bits_size; i++) 
            mask |= 1<<active_bits[i];
        inv_mask = ~mask;
        int active_bits_size_power = 1<<active_bits_size;
        uint32_t num = base;
        data.push_back(num);
        for(int i = 1; i < active_bits_size_power; i++) {
            num ^= 1<<active_bits[__builtin_ctz(i)];
            data.push_back(num);
        }
        std::sort(data.begin(), data.end());
    }
    ~window_space() {}

    int find(uint32_t x) {
        uint32_t index = std::lower_bound(data.begin(), data.end(), x) - data.begin();
        if(index < data.size() && data[index] == x) return index;
        return -1;
    }

    void save(std::string file_path) {
        FILE* f = fopen(file_path.c_str(), "wb+");
        for(auto c : data) {
            fwrite(&c, sizeof(uint32_t), 1, f);
        }
        fclose(f);
    }

    void load(std::string file_path, int size) {
        FILE* f = fopen(file_path.c_str(), "rb");
        data.clear();
        uint32_t tmp;
        for(int i = 0; i < size; i++) {
            fread(&tmp, sizeof(uint32_t), 1, f);
            data.push_back(tmp);
        }
        fclose(f);
    }
};



#endif