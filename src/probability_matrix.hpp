#ifndef __PROBABILITY_MATRIX__
#define __PROBABILITY_MATRIX__

#include <cstring>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <sys/mman.h>
#include <assert.h>
#include <sys/stat.h>
#include "window_space.hpp"
#include "hyperparameter.hpp"


int mkpath(std::string s,mode_t mode=0755) {
    size_t pre = 0, pos;
    std::string dir;
    int mdret = 0;
    if(s[s.size()-1]!='/') s+='/';
    while((pos=s.find_first_of('/',pre))!=std::string::npos){
        dir=s.substr(0,pos++);
        pre=pos;
        if(dir.size()==0) continue;
        if((mdret=::mkdir(dir.c_str(),mode)) && errno!=EEXIST){
            return mdret;
        }
    }
    return mdret;
}


void *alloc_distribution(uint32_t size) {
    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    assert(p != MAP_FAILED);
    return p;
}
void free_distribution(void *p, uint32_t size) { munmap(p, size); }


class probability_matrix {
public:
    window_space left, right;
    int left_size, right_size;
    probability* prob;

    probability_matrix(window_space left, window_space right) {
        this->left = left;
        this->right = right;
        left_size = left.data.size(), right_size = right.data.size();
        int all_size = left_size * right_size;
        prob = (probability*) alloc_distribution(sizeof(probability) * all_size);
        for(int i = 0; i < all_size; i++) prob[i] = 0;
    }
    void unalloc() {
        free_distribution(prob, sizeof(probability) * left_size * right_size);
        prob = NULL;
    }
    probability& operator() (const int l, const int r) const {
        return prob[l * right_size + r];
    }
    ~probability_matrix() {}

    void save(std::string floder, int round) {
        mkpath(floder);
        left.save(floder + "/left"), right.save(floder + "/right");
        std::string file_path = floder + "/prob";
        FILE* f = fopen(file_path.c_str(), "wb+");
        fwrite(prob, sizeof(probability), left_size * right_size, f);
        fclose(f);

        std::ofstream out(floder + "info.txt", std::ios::out);
        out << round << std::endl;
        out << left_size << " " << right_size << std::endl;
        out << left.base << " " << left.mask << " " << left.inv_mask << std::endl;
        out << right.base << " " << right.mask << " " << right.inv_mask << std::endl;
        out.close();
    }

    void load(std::string floder, int& round) {
        unalloc();

        std::ifstream in(floder + "info.txt", std::ios::in);
        in >> round; round++;
        in >> left_size >> right_size;
        in >> left.base >> left.mask >> left.inv_mask;
        in >> right.base >> right.mask >> right.inv_mask;
        in.close();

        prob = (probability*) alloc_distribution(sizeof(probability) * left_size * right_size);
        std::string file_path = floder + "/prob";
        FILE* f = fopen(file_path.c_str(), "rb");
        assert(fread(prob, sizeof(probability), left_size * right_size, f));
        fclose(f);
        left.load(floder + "/left", left_size);
        right.load(floder + "/right", right_size);
    }
};

#endif