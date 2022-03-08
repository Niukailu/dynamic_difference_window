#ifndef __SIMECK64_HPP__
#define __SIMECK64_HPP__

#include <stdint.h>


#define ROT(x, r) (((x) << (r)) | ((x) >> (32 - (r))))
inline void round64(uint32_t key, uint32_t& left, uint32_t& right) {
    uint32_t tmp = left;
    left = (left & ROT(left, 5)) ^ ROT(left, 1) ^ right ^ key;
    right = tmp;
}
inline void swap3(uint32_t& a, uint32_t& b, uint32_t& c) {
        uint32_t temp = a;
        a = b;
        b = c;
        c = temp;
}

void simeck_64_128(const uint32_t master_key[], const uint32_t plaintext[], uint32_t ciphertext[], int rounds=44) {
    uint32_t keys[4] = {master_key[0], master_key[1], master_key[2], master_key[3]};
    ciphertext[0] = plaintext[0], ciphertext[1] = plaintext[1];
    uint32_t constant = 0xFFFFFFFC;
    uint64_t sequence = 0x938BCA3083F;
    for (int idx = 0; idx < rounds; idx++) {
        round64(keys[0], ciphertext[1], ciphertext[0]);
        constant = (constant & 0xFFFFFFFC) | (sequence & 1);
        sequence >>= 1;
        round64(constant, keys[1], keys[0]);
        swap3(keys[1], keys[2], keys[3]);
    }
}

#endif  // __SIMECK64_HPP__