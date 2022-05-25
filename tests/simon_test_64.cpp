#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <time.h>
#include <bits/stdc++.h>
using namespace std;

#define u8 uint8_t
#define u32 uint32_t
#define u64 uint64_t

#define ROTL32(x,r) (((x)<<(r)) | (x>>(32-(r))))
#define ROTR32(x,r) (((x)>>(r)) | ((x)<<(32-(r))))

#define f32(x) ((ROTL32(x,1) & ROTL32(x,8)) ^ ROTL32(x,2))
#define R32x2(x,y,k1,k2) (y^=f32(x), y^=k1, x^=f32(y), x^=k2)

void Words32ToBytes(u32 words[],u8 bytes[],int numwords) {
    int i,j=0;
    for(i=0;i<numwords;i++){
        bytes[j]=(u8)words[i];
        bytes[j+1]=(u8)(words[i]>>8);
        bytes[j+2]=(u8)(words[i]>>16);
        bytes[j+3]=(u8)(words[i]>>24);
        j+=4;
    }
}

void BytesToWords32(u8 bytes[],u32 words[],int numbytes) {
    int i,j=0;
    for(i=0;i<numbytes/4;i++){
        words[i]=(u32)bytes[j] | ((u32)bytes[j+1]<<8) | ((u32)bytes[j+2]<<16) |
        ((u32)bytes[j+3]<<24); j+=4;
    }
}

void Simon64128KeySchedule(u32 K[],u32 rk[], int round)
{
    u32 i,c=0xfffffffc;
    u64 z=0xfc2ce51207a635dbLL;
    rk[0]=K[0]; rk[1]=K[1]; rk[2]=K[2]; rk[3]=K[3];
    for(i=4;i<round;i++){
        rk[i]=c^(z&1)^rk[i-4]^ROTR32(rk[i-1],3)^rk[i-3]
        ^ROTR32(rk[i-1],4)^ROTR32(rk[i-3],1);
        z>>=1;
    }
}

void Simon64128Encrypt(u32 Pt[],u32 Ct[],u32 rk[], int round) {
    u32 i;
    Ct[1]=Pt[1]; Ct[0]=Pt[0];
    for(i=0;i<round;) R32x2(Ct[1],Ct[0],rk[i++],rk[i++]);
}

void Simon64128Decrypt(u32 Pt[],u32 Ct[],u32 rk[], int round) {
    int i;
    Pt[1]=Ct[1]; Pt[0]=Ct[0];
    for(i=round-1;i>=0;) R32x2(Pt[0],Pt[1],rk[i--],rk[i--]);
}

void simon(u8 pt[8], u8 k[16], u32 Ct[2], int round){
    u32 Pt[2], K[4], rk[44];
    BytesToWords32(pt, Pt, 8);
    BytesToWords32(k, K, 16);
    Simon64128KeySchedule(K, rk, round);
    Simon64128Encrypt(Pt, Ct, rk, round);
}


int main() {
    srand(time(0));
    // int epochs = 1 << 28, num = 0, round = 10;
    // u32 delta_[2] = {0x40000004, 0x1}, ans[2] = {0x40000440, 0x1000};
    
    // int epochs = 1 << 27, num = 0, round = 10;
    // u32 delta_[2] = {0x1880, 0x440}, ans[2] = {0x10100, 0x44040};

    // int epochs = 1 << 26, num = 0, round = 10;
    // u32 delta_[2] = {0x222000, 0x80000}, ans[2] = {0x222000, 0x808000};
    
    int epochs = 1 << 28, num = 0, round = 10;
    u32 delta_[2] = {0x11000000, 0x4000000}, ans[2] = {0x1000011, 0x40};

    u8 pt1[8], pt2[8], k[16];
    u8 delta[8];
    Words32ToBytes(delta_, delta, 2);
    u32 Ct1[2], Ct2[2];
    mt19937 mtrand(time(0));
    for(int epoch = 1; epoch <= epochs; epoch++) {
        for(int i = 0; i < 8; i++) {
            pt1[i] = (u8)rand();
            pt2[i] = pt1[i] ^ delta[i];
        }
        for(int i = 0; i < 16; i++) k[i] = (u8)rand();
        simon(pt1, k, Ct1, round);
        simon(pt2, k, Ct2, round);
        if(((Ct1[0] ^ Ct2[0]) == ans[0]) && ((Ct1[1] ^ Ct2[1]) == ans[1])) num++;
    }
    cout << num << endl;
    cout << log2(num * 1.0 / epochs) << endl;
    return 0;
}