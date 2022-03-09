#include <iostream>
#include <vector>
#include <cmath>
#include <bitset>
#include "src/window_space.hpp"
#include "src/probability.hpp"
#include "src/probability_matrix.hpp"
#include "src/transition_space.hpp"
#include "src/hyperparameter.hpp"
using namespace std;


// 定义核心函数
#define ROT(x,n) (((uint32_t)(x)<<((n)&31)) | ((uint32_t)(x)>>((32-(n))&31)))
inline uint32_t f(uint32_t x) { return (ROT(x, 5) & x) ^ ROT(x, 1); }
uint32_t lowbit(uint32_t x){ return x&(-x); }

// 定义全局变量
transition_space Matrix[1U<<PRECISION];
probability edge_prob[1<<PRECISION];
probability free_prob[32];
double base_prob[32];

// 计算概率
void get_prob(window_space& x) {
    int data_size = x.data.size();
    for(int l = 0; l < data_size; l++) {
        uint32_t delta = x.data[l];
        uint32_t base = f(0) ^ f(delta);
        uint32_t A[32];
        for(int i = 0; i < 32; i++) A[i] = f(1<<i) ^ f(delta^(1<<i)) ^ base;
        for(int i = 0, p = 0; i < 32 && p < 32; i++, p++) {
            for(int j = i; j < 32; j++) {
                if(A[j] & (1<<p)) { swap(A[i], A[j]); break; }
            }
            if(!(A[i] & (1<<p))) {i--; continue;}
            for (int j = 0; j < 32; j++) {
                if(i == j) continue;
	            if (A[j] & (1<<p)) A[j] ^= A[i];
            }
            if (base & (1<<p)) base ^= A[i];
        }
        Matrix[l].num = 0;
        Matrix[l].proba = 1;
        Matrix[l].base = base;
        for(int i = 0; i < 32; i++) {
            if(A[i]) {
                Matrix[l].proba /= TWO;
                Matrix[l].A[Matrix[l].num++] = A[i];
            }
        }
    }
}

// 计算概率密集window
window_space get_dense_window(probability_matrix& now) {
    // 计算边缘概率
    for(size_t i = 0; i < now.left_size; i++) {
        edge_prob[i] = 0;
        for(size_t j = 0; j < now.right_size; j++)
            edge_prob[i] += now(i, j);
    }

    // 计算32位中每一个位置的活跃度
    int idx[32];
    for(int i = 0; i < 32; i++) idx[i] = i, base_prob[i] = 0, free_prob[i] = 0;

    // 首先计算base
    for(size_t i = 0; i < now.left_size; i++) {
        for(size_t j = 0; j < now.right_size; j++) {
            uint32_t br = now.right.data[j] ^ Matrix[i].base; // 计算 r^m[l].base
            for(int k = 0; k < 32; k++) {
                if(br & (1<<k)) base_prob[k] += now(i, j)(); //权重就是prob[l][r]
                else base_prob[k] -= now(i, j)();
            }
        }
    }
    uint32_t base = 0;
    for(int i = 31; i >= 0; i--) {
        base <<= 1;
        if(base_prob[i] > 0) base |= 1;  //得到base
    }

    // 下面计算活跃位
    for(uint32_t i = 0; i < now.left_size; i++) {       // for l
        uint32_t free = 0;
        int bits[32] = {0}; //统计A数组每个元素相应bit位的个数
        for(int j = 0; j < Matrix[i].num; j++) {        // for len( M[l].A )
            free |= Matrix[i].A[j];
            for(int k = 0; k < 32; k++) {
                if(Matrix[i].A[j] & (1<<k)) bits[k]++;  //若为活跃位则++
            }
        }
        for(int j = 0; j < 32; j++) bits[j] = bits[j] ? 1 : 0; // [*] repeat or no repeats
        // 统计 r ^ m[l].base ^ base的非0位(非0位要作为活跃位)
        for(size_t j = 0; j < now.right_size; j++) {    // for r
            uint32_t brbase = now.right.data[j] ^ Matrix[i].base ^ base;
            // brbase |= free;   // [*] free or no free
            for (int k = 0; k < 32 && brbase; k++) {
                if(brbase & (1<<k)) free_prob[k] += now(i, j) * Matrix[i].proba; //权重now(i, j) * Matrix[i].proba
            }
        }
        double all_bits_num = 0;
        for(int j = 0; j < 32; j++) all_bits_num += bits[j]; //为得到每位上的 (活跃位个数 / 总数) 这个占比
        for(int j = 0; j < 32 && all_bits_num; j++) free_prob[j] += edge_prob[i] * Matrix[i].proba * (bits[j]  / all_bits_num); //
    }
    // 根据概率对idx(下标)排序，得到活跃位
    sort(idx, idx + 32, [&](int a, int b) {
        return free_prob[a] == free_prob[b] ? a < b : free_prob[a] > free_prob[b];
    });
    vector<int> active_bits;
    for(int i = 0; i < PRECISION; i++) {
        if(free_prob[idx[i]].value != ZERO) active_bits.push_back(idx[i]);
    }
    // 得到窗口
    return window_space(base, active_bits);
}

// 根据window化简计算出来的概率
void simple_prob(window_space& x, window_space& dense_window) {
    int data_size = x.data.size();
    for(int l = 0; l < data_size; l++) {
        for(int i = 0, p = 31; i < Matrix[l].num && p >= 0; i++, p--) {
            if(!(dense_window.mask & (1<<p))) { i--; continue; }
            for(int j = i; j < Matrix[l].num; j++) {
                if(Matrix[l].A[j] & (1<<p)) {
                    swap(Matrix[l].A[i], Matrix[l].A[j]);
                    break;
                }
            }
            if(!(Matrix[l].A[i] & (1<<p))) { i--; continue; }
            for(int j = 0; j < Matrix[l].num; j++) {
                if(j == i) continue;
                if(Matrix[l].A[j] & (1<<p)) Matrix[l].A[j] ^= Matrix[l].A[i];
            }
            if(Matrix[l].base & (1<<p)) Matrix[l].base ^= Matrix[l].A[i];
        }
        Matrix[l].out_num = 0;
        for(int i = 0; i < Matrix[l].num; i++) {
            if(Matrix[l].A[i] & dense_window.inv_mask)
                swap(Matrix[l].A[Matrix[l].out_num++], Matrix[l].A[i]);
        }
    }
}

// 概率转移
probability_matrix round_trans(probability_matrix& now) {
    get_prob(now.left);
    window_space dense_window = get_dense_window(now);      // 获取概率密集区域
    printf("window is [");
    for(int i = 0; i < 32; i++) {
        if(dense_window.mask & (1<<i)) printf("%d, ", i);
    }
    printf("]\n");
    fflush(stdout);
    probability_matrix next(dense_window, now.left);
    simple_prob(now.left, dense_window);        // 化简概率

    for(uint32_t l = 0; l < now.left_size; l++) {
        for(uint32_t r = 0; r < now.right_size; r++) {
            probability nowlr = now(l, r);
            if(nowlr == 0) continue;
            transition_space m = Matrix[l];
            uint32_t d = now.right.data[r] ^ m.base;
            uint32_t dd = (d ^ dense_window.base) & dense_window.inv_mask;
            // 使用out的基向量化简dd和d，使之尽可能落在window内
            for(int i = 0; i < m.out_num && dd; i++) {
                if(((dd ^ m.A[i]) & dense_window.inv_mask) < dd) dd ^= m.A[i], d ^= m.A[i];
            }
            if(dd) continue; // 如果还是有window外的活跃位，那就真的没办法了
            probability nowlr_m_proba = nowlr * m.proba;
            for (int i=1;; i++) {
                int indexd = dense_window.find(d);
                next(indexd, l) += nowlr_m_proba;
                int z = __builtin_ctz(i);
                if (z < m.num - m.out_num) d ^= m.A[z + m.out_num];
                else break;
            }
        }
    }
    now.unalloc();
    return next;
}

void print(probability_matrix& now) {
    probability max_prob = 0;
    for (uint32_t i = 0; i < now.left_size; i++) {
	    for (uint32_t j = 0; j < now.right_size; j++) {
            if(now(i, j) > max_prob) max_prob = now(i, j);
	    }
    }
    printf("Max: %.6f", max_prob.value);
    for (uint32_t i = 0; i < now.left_size; i++) {
	    for (uint32_t j = 0; j < now.right_size; j++) {
	        if (now(i, j) > max_prob*zo6) {
	            printf (" (%#x,%#x)", now.left.data[i], now.right.data[j]);
	        }
	    }
    }
    printf ("\n");
}

int main() {
    int begin_round = 1;
    window_space left(begin_left, {}), right(begin_right, {});
    probability_matrix now(left, right);
    now(now.left.find(begin_left), now.right.find(begin_right)) = 1;       // now[0, 1] = 1

    if (LOAD_ROUND > 0) {   // 加载上次结果
        now.load("experiments/SIMECK_" + name + "/" + to_string(LOAD_ROUND) + "/", begin_round);
        cout << "第" << begin_round - 1 << "轮结果被成功加载！" << endl;
    }
    now.save("experiments/SIMECK_" + name + "/" + to_string(begin_round - 1) + "/", begin_round - 1);

    printf("Input: (%#x, %#x) -> (%#x, %#x)\n", begin_left, begin_right, end_left, end_right);
    for (int i = begin_round; i < ROUNDS; i++) {
        print(now);
        fflush(stdout);
        now = round_trans(now);
        now.save("experiments/SIMECK_" + name + "/" + to_string(i) + "/", i);
        int left_index = now.left.find(end_left), right_index = now.right.find(end_right);
        if(left_index == -1 || right_index == -1) printf("Round %2i: -inf\n", i);
        else printf("Round %2i: %f\n", i, now(left_index, right_index).value);
        fflush(stdout);
    }
    now.unalloc();

    return 0;
}