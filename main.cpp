#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <vector>
#include <cmath>
#include <bitset>
#include "src/window_space.hpp"
#include "src/probability.hpp"
#include "src/probability_matrix.hpp"
#include "src/transition_space.hpp"
#include "src/hyperparameter.hpp"
using namespace std;


// 定义全局变量
FILE* log_fp;
transition_space Matrix[1U<<PRECISION];
probability edge_prob[1<<PRECISION];
probability free_prob[BITS];
double base_prob[BITS];

// 计算概率
void get_prob(window_space& x) {
    int data_size = x.data.size();
    for(int l = 0; l < data_size; l++) {
        dtype delta = x.data[l];
        dtype base = f(0) ^ f(delta);
        dtype A[BITS];
        for(int i = 0; i < BITS; i++) A[i] = f(1<<i) ^ f(delta^(1<<i)) ^ base;
        for(int i = 0, p = 0; i < BITS && p < BITS; i++, p++) {
            for(int j = i; j < BITS; j++) {
                if(A[j] & (1<<p)) { swap(A[i], A[j]); break; }
            }
            if(!(A[i] & (1<<p))) {i--; continue;}
            for (int j = 0; j < BITS; j++) {
                if(i == j) continue;
	            if (A[j] & (1<<p)) A[j] ^= A[i];
            }
            if (base & (1<<p)) base ^= A[i];
        }
        Matrix[l].num = 0;
        Matrix[l].proba = 1;
        Matrix[l].base = base;
        for(int i = 0; i < BITS; i++) {
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
    for(int i = 0; i < now.left_size; i++) {
        edge_prob[i] = 0;
        for(int j = 0; j < now.right_size; j++)
            edge_prob[i] += now(i, j);
    }

    // 计算{{BITS}}位中每一个位置的活跃度
    int idx[BITS];
    for(int i = 0; i < BITS; i++) idx[i] = i, base_prob[i] = 0, free_prob[i] = 0;

    // 首先计算base
    for(int i = 0; i < now.left_size; i++) {
        for(int j = 0; j < now.right_size; j++) {
            dtype br = now.right.data[j] ^ Matrix[i].base; // 计算 r^m[l].base
            for(int k = 0; k < BITS; k++) {
                if(br & (1<<k)) base_prob[k] += now(i, j)(); //权重就是prob[l][r]
                else base_prob[k] -= now(i, j)();
            }
        }
    }
    dtype base = 0;
    for(int i = BITS - 1; i >= 0; i--) {
        base <<= 1;
        if(base_prob[i] > 0) base |= 1;  //得到base
    }

    // 下面计算活跃位
    for(int i = 0; i < now.left_size; i++) {       // for l
        dtype free = 0;
        int bits[BITS] = {0}; //统计A数组每个元素相应bit位的个数
        for(int j = 0; j < Matrix[i].num; j++) {        // for len( M[l].A )
            free |= Matrix[i].A[j];
            for(int k = 0; k < BITS; k++) {
                if(Matrix[i].A[j] & (1<<k)) bits[k]++;  //若为活跃位则++
            }
        }
        for(int j = 0; j < BITS; j++) bits[j] = bits[j] ? 1 : 0; // [*] repeat or no repeats
        // 统计 r ^ m[l].base ^ base的非0位(非0位要作为活跃位)
        for(int j = 0; j < now.right_size; j++) {    // for r
            dtype brbase = now.right.data[j] ^ Matrix[i].base ^ base;
            // brbase |= free;   // [*] free or no free
            for (int k = 0; k < BITS && brbase; k++) {
                if(brbase & (1<<k)) free_prob[k] += now(i, j) * Matrix[i].proba; //权重now(i, j) * Matrix[i].proba
            }
        }
        double all_bits_num = 0;
        for(int j = 0; j < BITS; j++) all_bits_num += bits[j]; //为得到每位上的 (活跃位个数 / 总数) 这个占比
        for(int j = 0; j < BITS && all_bits_num; j++) free_prob[j] += edge_prob[i] * Matrix[i].proba * (bits[j]  / all_bits_num); //
    }
    // 根据概率对idx(下标)排序，得到活跃位
    sort(idx, idx + BITS, [&](int a, int b) {
        return free_prob[a] == free_prob[b] ? a < b : free_prob[a] > free_prob[b];
    });
    vector<int> active_bits;
    for(int i = 0; i < PRECISION; i++) {
        if(free_prob[idx[i]].value != ZERO) active_bits.push_back(idx[i]);
    }
    // 得到窗口
    return window_space(base, active_bits);
}

//删除窗口外的A
// 根据window化简计算出来的概率
void simple_prob(window_space& x, window_space& dense_window) {
    int data_size = x.data.size();
    for(int l = 0; l < data_size; l++) {
        for(int i = 0, p = BITS - 1; i < Matrix[l].num && p >= 0; i++, p--) {
            if(dense_window.mask & (1<<p)) { i--; continue; }
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
    printf("window is ["), fprintf(log_fp, "window is [");
    for(int i = 0; i < BITS; i++) {
        if(dense_window.mask & (1<<i)) printf("%d, ", i), fprintf(log_fp, "%d, ", i);
    }
    printf("]\n"), fprintf(log_fp, "]\n");
    fflush(stdout);
    probability_matrix next(dense_window, now.left);
    simple_prob(now.left, dense_window);        // 化简概率

    for(int l = 0; l < now.left_size; l++) {
        for(int r = 0; r < now.right_size; r++) {
            probability nowlr = now(l, r);
            if(nowlr == 0) continue;
            transition_space m = Matrix[l];
            dtype d = now.right.data[r] ^ m.base;
            dtype dd = (d ^ dense_window.base) & dense_window.inv_mask;
            // 使用out的基向量化简dd和d，使之尽可能落在window内
            for(int i = 0; i < m.out_num && dd; i++) {
                if(((dd ^ m.A[i]) & dense_window.inv_mask) < dd) dd ^= m.A[i], d ^= m.A[i];
            }
            if(dd & dense_window.inv_mask) continue; // 如果还是有window外的活跃位，那就真的没办法了
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
    for (int i = 0; i < now.left_size; i++) {
	    for (int j = 0; j < now.right_size; j++) {
            if(now(i, j) > max_prob) max_prob = now(i, j);
	    }
    }
    printf("Max: %.6f", max_prob.value), fprintf(log_fp, "Max: %.6f", max_prob.value);
    for (int i = 0; i < now.left_size; i++) {
	    for (int j = 0; j < now.right_size; j++) {
	        if (now(i, j) > max_prob*zo6) {
	            printf(" (%#x,%#x)", now.left.data[i], now.right.data[j]), fprintf(log_fp, " (%#x,%#x)", now.left.data[i], now.right.data[j]);
	        }
	    }
    }
    printf("\n"), fprintf(log_fp, "\n");
}

int main() {
    int begin_round = 1;
    window_space left(begin_left, {}), right(begin_right, {});
    probability_matrix now(left, right);
    now(now.left.find(begin_left), now.right.find(begin_right)) = 1;       // now[0, 1] = 1

    // 创建文件夹
    mkpath("experiments/");
    mkpath("experiments/" + name + "/");

    if (LOAD_ROUND > 0) {   // 加载上次结果
        log_fp = fopen(("experiments/" + name + "/info.log").c_str(), "a+");    // 配置输出
        now.load("experiments/" + name + "/" + to_string(LOAD_ROUND) + "/", begin_round);
        printf("第%d轮结果被成功加载！\n", begin_round - 1);
    }
    else{
        log_fp = fopen(("experiments/" + name + "/info.log").c_str(), "w+");    // 配置输出
        now.save("experiments/" + name + "/" + to_string(begin_round - 1) + "/", begin_round - 1);
        printf("Input: (%#x, %#x) -> (%#x, %#x)\n", begin_left, begin_right, end_left, end_right);
        fprintf(log_fp, "Input: (%#x, %#x) -> (%#x, %#x)\n", begin_left, begin_right, end_left, end_right);
    }
    for (int i = begin_round; i < ROUNDS; i++) {
        print(now);
        fflush(stdout), fflush(log_fp);
        now = round_trans(now);
        now.save("experiments/" + name + "/" + to_string(i) + "/", i);
        int left_index = now.left.find(end_left), right_index = now.right.find(end_right);
        if(left_index == -1 || right_index == -1) {
            printf("Round %2i: -inf\n", i);
            fprintf(log_fp, "Round %2i: -inf\n", i);
        }
        else{
            printf("Round %2i: %f\n", i, now(left_index, right_index).value);
            fprintf(log_fp, "Round %2i: %f\n", i, now(left_index, right_index).value);
        }
        fflush(stdout), fflush(log_fp);
    }
    print(now);
    fflush(stdout), fflush(log_fp);

    now.unalloc();
    fclose(log_fp);

    return 0;
}