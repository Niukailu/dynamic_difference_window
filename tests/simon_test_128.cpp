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

#define ROTL64(x,r) (((x)<<(r)) | (x>>(64-(r))))
#define ROTR64(x,r) (((x)>>(r)) | ((x)<<(64-(r))))

#define f64(x) ((ROTL64(x,1) & ROTL64(x,8)) ^ ROTL64(x,2))
#define R64x2(x,y,k1,k2) (y^=f64(x), y^=k1, x^=f64(y), x^=k2)

void Words64ToBytes(u64 words[],u8 bytes[],int numwords) {
    int i,j=0;
    for(i=0;i<numwords;i++){
        bytes[j]=(u8)words[i];
        bytes[j+1]=(u8)(words[i]>>8);
        bytes[j+2]=(u8)(words[i]>>16);
        bytes[j+3]=(u8)(words[i]>>24);
        bytes[j+4]=(u8)(words[i]>>32);
        bytes[j+5]=(u8)(words[i]>>40);
        bytes[j+6]=(u8)(words[i]>>48);
        bytes[j+7]=(u8)(words[i]>>56);
        j+=8;
    }
}


void BytesToWords64(u8 bytes[],u64 words[],int numbytes)
{
    int i,j=0;
    for(i=0;i<numbytes/8;i++){
        words[i]=(u64)bytes[j] | ((u64)bytes[j+1]<<8) | ((u64)bytes[j+2]<<16) |
        ((u64)bytes[j+3]<<24) | ((u64)bytes[j+4]<<32) | ((u64)bytes[j+5]<<40) |
        ((u64)bytes[j+6]<<48) | ((u64)bytes[j+7]<<56); j+=8;
    }
}

void Simon128256KeySchedule(u64 K[],u64 rk[], int round) {
    u64 i,D=K[3],C=K[2],B=K[1],A=K[0];
    u64 c=0xfffffffffffffffcLL, z=0xfdc94c3a046d678bLL;
    for(i=0;i<round;){
        rk[i++]=A; A^=c^(z&1)^ROTR64(D,3)^ROTR64(D,4)^B^ROTR64(B,1); z>>=1;
        rk[i++]=B; B^=c^(z&1)^ROTR64(A,3)^ROTR64(A,4)^C^ROTR64(C,1); z>>=1;
        rk[i++]=C; C^=c^(z&1)^ROTR64(B,3)^ROTR64(B,4)^D^ROTR64(D,1); z>>=1;
        rk[i++]=D; D^=c^(z&1)^ROTR64(C,3)^ROTR64(C,4)^A^ROTR64(A,1); z>>=1;
    }
}


void Simon128256Encrypt(u64 Pt[],u64 Ct[],u64 rk[], int round)
{
    u64 i;
    Ct[0]=Pt[0]; Ct[1]=Pt[1];
    for(i=0;i<round;i+=2) R64x2(Ct[1],Ct[0],rk[i],rk[i+1]);
}
void Simon128256Decrypt(u64 Pt[],u64 Ct[],u64 rk[])
{
    int i;
    Pt[0]=Ct[0]; Pt[1]=Ct[1];
    for(i=71;i>=0;i-=2) R64x2(Pt[0],Pt[1],rk[i],rk[i-1]);
}


void simon(u8 pt[16], u8 k[32], u64 Ct[2], int round){
    u64 Pt[2], K[4], rk[44];
    BytesToWords64(pt, Pt, 16);
    BytesToWords64(k, K, 32);
    Simon128256KeySchedule(K, rk, round);
    Simon128256Encrypt(Pt, Ct, rk, round);
}


int main() {
    srand(time(0));
    int epochs = 1 << 25, num = 0, round = 10;
    u64 delta_[2] = {0x4440, 0x1000}, ans[2] = {0x4440, 0x10100};

    u8 pt1[16], pt2[16], k[32];
    u8 delta[16];
    Words64ToBytes(delta_, delta, 2);
    u64 Ct1[2], Ct2[2];
    for(int epoch = 1; epoch <= epochs; epoch++) {
        for(int i = 0; i < 16; i++) {
            pt1[i] = (u8)rand();
            pt2[i] = pt1[i] ^ delta[i];
        }
        for(int i = 0; i < 32; i++) k[i] = (u8)rand();
        simon(pt1, k, Ct1, round);
        simon(pt2, k, Ct2, round);
        if(((Ct1[0] ^ Ct2[0]) == ans[0]) && ((Ct1[1] ^ Ct2[1]) == ans[1])) num++;
    }
    cout << num << endl;
    cout << log2(num * 1.0 / epochs) << endl;
    return 0;
}