// Runtime-specialized CUDA kernels. All indices and cipher words are 64-bit.
typedef unsigned long long U;
__device__ U rot(U x, int r) {
    r = (r % BITS + BITS) % BITS;
    return r ? ((x << r) | (x >> (BITS-r))) & WORD_MASK : x;
}
__device__ U fun(U x) {
    return (rot(x, RA) & rot(x, RB)) ^ rot(x, RC);
}
extern "C" __global__ void basis(const U* left, U count, U* bases,
                                  U* vectors, int* ranks, U* supports) {
    U l = (U)blockIdx.x * blockDim.x + threadIdx.x;
    if(l >= count) return;
    U delta = left[l], a[BITS];
    U base = LINEAR_MODE ? rot(delta, -RC) : fun(delta);
    for(int j=0;j<BITS;j++) {
        U e = 1ULL << j;
        a[j] = LINEAR_MODE ? rot((delta & rot(e, RA-RB)) ^ rot(delta & e, RB-RA), -RB)
                          : fun(e) ^ fun(delta ^ e) ^ base;
    }
    int rank=0;
    for(int p=0;p<BITS;p++) {
        int k=rank;
        while(k<BITS && !(a[k] & (1ULL<<p))) k++;
        if(k==BITS) continue;
        U tmp=a[rank]; a[rank]=a[k]; a[k]=tmp;
        for(int j=0;j<BITS;j++) if(j!=rank && (a[j] & (1ULL<<p))) a[j]^=a[rank];
        if(base & (1ULL<<p)) base ^= a[rank];
        rank++;
    }
    U support=0;
    for(int j=0;j<BITS;j++) { vectors[l*BITS+j]=a[j]; support |= a[j]; }
    bases[l]=base; ranks[l]=rank; supports[l]=support;
}
extern "C" __global__ void restrict_basis(U count, U mask, U* bases, U* vectors,
                                           const int* ranks, int* outside) {
    U l=(U)blockIdx.x*blockDim.x+threadIdx.x;
    if(l>=count) return;
    U a[BITS], base=bases[l]; int rank=ranks[l], out=0;
    for(int j=0;j<rank;j++) a[j]=vectors[l*BITS+j];
    for(int p=BITS-1;p>=0 && out<rank;p--) {
        if(mask & (1ULL<<p)) continue;
        int k=out;
        while(k<rank && !(a[k] & (1ULL<<p))) k++;
        if(k==rank) continue;
        U tmp=a[out]; a[out]=a[k]; a[k]=tmp;
        for(int j=0;j<rank;j++) if(j!=out && (a[j] & (1ULL<<p))) a[j]^=a[out];
        if(base & (1ULL<<p)) base ^= a[out];
        out++;
    }
    for(int j=0;j<rank;j++) vectors[l*BITS+j]=a[j];
    bases[l]=base; outside[l]=out;
}
__device__ U pack(U x, U mask) {
    U result=0, bit=1;
    while(mask) { U low=mask & -mask; if(x & low) result |= bit; bit <<= 1; mask ^= low; }
    return result;
}
extern "C" __global__ void pack_basis(U count, U mask, const U* vectors,
    U* packed) {
    U i=(U)blockIdx.x*blockDim.x+threadIdx.x;
    U stride=(U)gridDim.x*blockDim.x;
    for(;i<count*BITS;i+=stride) packed[i]=pack(vectors[i],mask);
}
extern "C" __global__ void propagate(const double* prob, U nl, U nr, const U* right,
    const U* bases, const U* vectors, const U* packed, const int* ranks, const int* outside,
    U mask, U base, double* next) {
    U index=(U)blockIdx.x*blockDim.x+threadIdx.x;
    U stride=(U)gridDim.x*blockDim.x;
    for(;index<nl*nr;index+=stride) {
        double weight=prob[index]; if(weight==0) continue;
        U l=index/nr, r=index%nr;
        U d=right[r]^bases[l], dd=(d^base)&~mask;
        int out=outside[l], rank=ranks[l];
        for(int j=0;j<out && dd;j++) {
            U a=vectors[l*BITS+j];
            if(((dd^a)&~mask)<dd) {dd=(dd^a)&~mask; d^=a;}
        }
        if(dd & ~mask) continue;
        weight=ldexp(weight,-rank);
        U coordinate=pack(d,mask);
        U total=1ULL<<(rank-out);
        for(U i=0;i<total;i++) {
            atomicAdd(next+coordinate*nl+l,weight);
            if(i+1<total) coordinate ^= packed[l*BITS+out+__ffsll(i+1)-1];
        }
    }
}
// Pull formulation: each thread owns one output cell. Intersect the affine
// transition space with the INPUT right window, then sum its preimages.
// No atomic floating-point updates, and no binary search.
extern "C" __global__ void gather(const double* prob, U nl, U nr, U nt,
    const U* target, const double* row_mass, const U* bases, const U* vectors,
    const U* packed, const int* ranks, const int* outside, U mask, U base, double* next) {
    U index=(U)blockIdx.x*blockDim.x+threadIdx.x;
    U stride=(U)gridDim.x*blockDim.x;
    for(;index<nt*nl;index+=stride) {
        U l=index/nt, t=index%nt;
        U destination=t*nl+l;
        if(row_mass[l]==0) { next[destination]=0; continue; }
        U r=target[t]^bases[l], dd=(r^base)&~mask;
        int out=outside[l], rank=ranks[l];
        for(int j=0;j<out && dd;j++) {
            U a=vectors[l*BITS+j];
            if(((dd^a)&~mask)<dd) {dd=(dd^a)&~mask; r^=a;}
        }
        if(dd & ~mask) { next[destination]=0; continue; }
        U coordinate=pack(r,mask);
        U total=1ULL<<(rank-out);
        double sum=0;
        for(U i=0;i<total;i++) {
            sum += prob[l*nr+coordinate];
            if(i+1<total) coordinate ^= packed[l*BITS+out+__ffsll(i+1)-1];
        }
        next[destination]=ldexp(sum,-rank);
    }
}
// Quotient-space algorithm: aggregate each input row into cosets of
// H = transition image intersect input-right window. Every output in one coset
// has the same preimage sum. Work is linear in the matrix size, not 2^dim(H).
extern "C" __global__ void coset_plan(U nl, U nr, const U* packed,
    const int* ranks, const int* outside, const double* mass,
    U* reducers, U* quotient_mask, U* sizes) {
    U l=(U)blockIdx.x*blockDim.x+threadIdx.x;
    if(l>=nl) return;
    U a[20]; int h=ranks[l]-outside[l];
    for(int j=0;j<h;j++) a[j]=packed[l*BITS+outside[l]+j];
    U mask=nr-1; int k=0;
    for(int p=19;p>=0 && k<h;p--) {
        int j=k;
        while(j<h && !(a[j]&(1ULL<<p))) j++;
        if(j==h) continue;
        U tmp=a[k]; a[k]=a[j]; a[j]=tmp;
        for(int t=0;t<h;t++) if(t!=k && (a[t]&(1ULL<<p))) a[t]^=a[k];
        mask &= ~(1ULL<<p); k++;
    }
    for(int j=0;j<h;j++) reducers[l*20+j]=a[j];
    quotient_mask[l]=mask;
    sizes[l]=mass[l]==0 ? 0 : nr>>h;
}
__device__ U quotient(U coordinate, U l, const U* reducers, int h, U mask) {
    for(int j=0;j<h;j++) {
        U v=reducers[l*20+j];
        if((coordinate^v)<coordinate) coordinate^=v;
    }
    return pack(coordinate,mask);
}
extern "C" __global__ void coset_sum(const double* prob, U nl, U nr,
    const int* ranks, const int* outside, const U* reducers,
    const U* quotient_mask, const U* offsets, double* sums) {
    U index=(U)blockIdx.x*blockDim.x+threadIdx.x;
    U stride=(U)gridDim.x*blockDim.x;
    for(;index<nl*nr;index+=stride) {
        double value=prob[index]; if(value==0) continue;
        U l=index/nr, r=index%nr;
        U bucket=quotient(r,l,reducers,ranks[l]-outside[l],quotient_mask[l]);
        atomicAdd(sums+offsets[l]+bucket,value);
    }
}
extern "C" __global__ void coset_lookup(U nl, U nt, const U* target,
    const double* mass, const U* bases, const U* vectors,
    const int* ranks, const int* outside, U mask, U base,
    const U* reducers, const U* quotient_mask, const U* offsets,
    const double* sums, double* next) {
    U index=(U)blockIdx.x*blockDim.x+threadIdx.x;
    U stride=(U)gridDim.x*blockDim.x;
    for(;index<nt*nl;index+=stride) {
        U l=index%nl, t=index/nl;
        if(mass[l]==0) { next[index]=0; continue; }
        U r=target[t]^bases[l], dd=(r^base)&~mask;
        int out=outside[l], rank=ranks[l];
        for(int j=0;j<out && dd;j++) {
            U a=vectors[l*BITS+j];
            if(((dd^a)&~mask)<dd) {dd=(dd^a)&~mask; r^=a;}
        }
        if(dd & ~mask) {next[index]=0; continue;}
        U bucket=quotient(pack(r,mask),l,reducers,rank-out,quotient_mask[l]);
        next[index]=ldexp(sums[offsets[l]+bucket],-rank);
    }
}
