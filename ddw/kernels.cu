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
// Compile the affine preimage/coset map into a linear map of target coordinates.
// The low q bits encode the bucket, higher bits encode unsatisfied constraints.
extern "C" __global__ void affine_columns(U nl, int width, const int* bits,
    U target_base, U right_mask, U right_base, const U* bases, const U* vectors,
    const int* ranks, const int* outside, const U* reducers,
    const U* quotient_mask, U* columns) {
    U i=(U)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nl*(width+1)) return;
    U l=i%nl; int column=i/nl, out=outside[l];
    U x=column==width ? target_base^bases[l]^right_base : 1ULL<<bits[column];
    U dd=x&~right_mask, residual_mask=WORD_MASK&~right_mask;
    for(int j=0;j<out;j++) {
        U a=vectors[l*BITS+j], projected=a&~right_mask;
        residual_mask &= ~(1ULL<<(63-__clzll(projected)));
        if(((dd^a)&~right_mask)<dd) {dd=(dd^a)&~right_mask; x^=a;}
    }
    U bucket=quotient(pack(x,right_mask),l,reducers,ranks[l]-out,quotient_mask[l]);
    columns[i]=(pack(dd,residual_mask)<<__popcll(quotient_mask[l]))|bucket;
}
extern "C" __global__ void affine_tables(U nl, int width, int chunks,
    const U* columns, U* tables) {
    U i=(U)blockIdx.x*blockDim.x+threadIdx.x;
    U stride=(U)gridDim.x*blockDim.x;
    for(;i<nl*256*chunks;i+=stride) {
        U l=i%nl; int value=(i/nl)%256, chunk=i/(nl*256);
        U key=chunk==0 ? columns[(U)width*nl+l] : 0;
        for(int b=0;b<8 && chunk*8+b<width;b++)
            if(value&(1<<b)) key^=columns[(U)(chunk*8+b)*nl+l];
        tables[i]=key;
    }
}
extern "C" __global__ void coset_lookup_lut(U nl, U nt, int chunks,
    const U* tables, const U* sizes, const U* offsets, const int* ranks,
    const double* sums, double* next) {
    U i=(U)blockIdx.x*blockDim.x+threadIdx.x;
    U stride=(U)gridDim.x*blockDim.x;
    for(;i<nl*nt;i+=stride) {
        U l=i%nl, t=i/nl;
        if(!sizes[l]) {next[i]=0; continue;}
        U key=0;
        for(int c=0;c<chunks;c++) key^=tables[((U)c*256+((t>>(c*8))&255))*nl+l];
        double scale=__longlong_as_double(((U)(1023-ranks[l]))<<52);
        next[i]=key<sizes[l] ? sums[offsets[l]+key]*scale : 0;
    }
}
// One input read produces row mass and all coordinate bit marginals.
// A block handles 1024 values: bits 0..9 are local, upper bits are tile constants.
extern "C" __global__ void bit_marginal_tiles(const double* prob, U nr,
    U tiles, double* partial) {
    U tile=blockIdx.x, row=tile/tiles, start=(tile%tiles)*1024;
    int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
    double v[4];
    for(int j=0;j<4;j++) {
        U r=start+tid+j*256;
        v[j]=r<nr ? prob[row*nr+r] : 0;
    }
    double all=(v[0]+v[1])+(v[2]+v[3]);
    __shared__ double warp_sums[11][8];
    for(int k=0;k<11;k++) {
        double x=k==10 ? all : k==9 ? v[2]+v[3] : k==8 ? v[1]+v[3]
                     : ((tid>>k)&1) ? all : 0;
        for(int d=16;d;d>>=1) x+=__shfl_down_sync(0xffffffff,x,d);
        if(lane==0) warp_sums[k][warp]=x;
    }
    __syncthreads();
    if(tid<11) {
        double x=0;
        for(int w=0;w<8;w++) x+=warp_sums[tid][w];
        partial[tile*11+tid]=x;
    }
}
extern "C" __global__ void bit_marginal_finish(U nl, U tiles, int width,
    const double* partial, double* marginal) {
    U i=(U)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nl*(width+1)) return;
    U l=i/(width+1); int k=i%(width+1);
    double total=0;
    for(U t=0;t<tiles;t++) {
        if(k==width) total+=partial[(l*tiles+t)*11+10];
        else if(k<10) total+=partial[(l*tiles+t)*11+k];
        else if((t>>(k-10))&1) total+=partial[(l*tiles+t)*11+10];
    }
    marginal[i]=total;
}
extern "C" __global__ void bit_marginal_rows(const double* prob, U nr,
    int width, double* marginal) {
    U l=blockIdx.x; int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
    double upper[12], total=0;
    #pragma unroll
    for(int k=0;k<12;k++) upper[k]=0;
    for(U r=tid;r<nr;r+=256) {
        double x=prob[l*nr+r]; if(x==0) continue; total+=x;
        #pragma unroll
        for(int k=0;k<12;k++) if((r>>(k+8))&1) upper[k]+=x;
    }
    __shared__ double warp_sums[21][8];
    #pragma unroll
    for(int k=0;k<21;k++) if(k<=width) {
        double x=k==width ? total : k<8 ? ((tid>>k)&1 ? total : 0) : upper[k-8];
        for(int d=16;d;d>>=1) x+=__shfl_down_sync(0xffffffff,x,d);
        if(lane==0) warp_sums[k][warp]=x;
    }
    __syncthreads();
    if(tid<=width) {
        double x=0;
        for(int w=0;w<8;w++) x+=warp_sums[tid][w];
        marginal[l*(width+1)+tid]=x;
    }
}
// Fused maximum / first argmax / sum: one pass over the probability matrix.
__device__ void combine_stats(double &maximum, U &index, double &sum) {
    for(int d=16;d;d>>=1) {
        double other=__shfl_down_sync(0xffffffff,maximum,d);
        U other_index=__shfl_down_sync(0xffffffff,index,d);
        double other_sum=__shfl_down_sync(0xffffffff,sum,d);
        if(other>maximum || (other==maximum && other_index<index)) {maximum=other; index=other_index;}
        sum+=other_sum;
    }
}
extern "C" __global__ void distribution_stats(const double* prob, U count,
    double* maxima, U* indices, double* sums) {
    U i=(U)blockIdx.x*blockDim.x+threadIdx.x, stride=(U)gridDim.x*blockDim.x;
    double maximum=-1, sum=0; U index=~0ULL;
    for(;i<count;i+=stride) {
        double x=prob[i]; sum+=x;
        if(x>maximum || (x==maximum && i<index)) {maximum=x; index=i;}
    }
    combine_stats(maximum,index,sum);
    __shared__ double wm[8], ws[8]; __shared__ U wi[8];
    int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    if(!lane) {wm[warp]=maximum; wi[warp]=index; ws[warp]=sum;}
    __syncthreads();
    if(warp==0) {
        maximum=lane<8?wm[lane]:-1; index=lane<8?wi[lane]:~0ULL; sum=lane<8?ws[lane]:0;
        combine_stats(maximum,index,sum);
        if(!lane) {maxima[blockIdx.x]=maximum; indices[blockIdx.x]=index; sums[blockIdx.x]=sum;}
    }
}
extern "C" __global__ void finish_stats(U count, const double* maxima,
    const U* indices, const double* sums, double* result) {
    double maximum=-1,sum=0; U index=~0ULL;
    for(U i=threadIdx.x;i<count;i+=256) {
        double x=maxima[i]; sum+=sums[i];
        if(x>maximum || (x==maximum && indices[i]<index)) {maximum=x; index=indices[i];}
    }
    combine_stats(maximum,index,sum);
    __shared__ double wm[8],ws[8]; __shared__ U wi[8];
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    if(!lane) {wm[warp]=maximum; wi[warp]=index; ws[warp]=sum;}
    __syncthreads();
    if(warp==0) {
        maximum=lane<8?wm[lane]:-1; index=lane<8?wi[lane]:~0ULL; sum=lane<8?ws[lane]:0;
        combine_stats(maximum,index,sum);
        if(!lane) {result[0]=maximum; result[1]=sum; result[2]=(double)index;}
    }
}
__device__ U deposit(U coordinate, U mask) {
    U result=0;
    while(mask) { U bit=mask&-mask; if(coordinate&1) result|=bit; coordinate>>=1; mask^=bit; }
    return result;
}
// Each subgroup owns one disjoint coset. Read each input exactly once, then
// reduce in registers: no floating-point atomic operations or bucket clearing.
extern "C" __global__ void coset_reduce(const double* prob, U nr,
    const int* ranks, const int* outside, const U* reducers,
    const U* quotient_mask, const U* offsets, const U* sizes, double* sums) {
    U l=blockIdx.x; int h=ranks[l]-outside[l], low=h<5?h:5;
    int group=1<<low, lane=threadIdx.x&(group-1);
    U count=1ULL<<h, base=offsets[l];
    for(U bucket=threadIdx.x/group;bucket<sizes[l];bucket+=256/group) {
        U coordinate=deposit(bucket,quotient_mask[l]);
        for(int j=0;j<low;j++) if(lane&(1<<j)) coordinate^=reducers[l*20+j];
        double sum=0;
        U iterations=count/group;
        for(U j=0;j<iterations;j++) {
            sum+=prob[l*nr+coordinate];
            if(j+1<iterations) coordinate^=reducers[l*20+low+__ffsll(j+1)-1];
        }
        unsigned active=__activemask();
        for(int d=group/2;d;d>>=1) sum+=__shfl_down_sync(active,sum,d,group);
        if(!lane) sums[base+bucket]=sum;
    }
}
// Reuse a reduction tree to compute all bit marginals in O(tile size) adds.
extern "C" __global__ void hierarchical_marginal_tiles(const double* prob,
    U nr, U tiles, double* partial) {
    U tile=blockIdx.x, row=tile/tiles, start=(tile%tiles)*4096;
    int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
    __shared__ double values[4096], warp_sums[8];
    for(int i=tid;i<4096;i+=256) values[i]=start+i<nr ? prob[row*nr+start+i] : 0;
    __syncthreads();
    for(int bit=0;bit<12;bit++) {
        double odd=0;
        for(int i=tid;i<(4096>>(bit+1));i+=256) {
            int even=i<<(bit+1);
            double b=values[even+(1<<bit)];
            values[even]+=b; odd+=b;
        }
        for(int d=16;d;d>>=1) odd+=__shfl_down_sync(0xffffffff,odd,d);
        if(!lane) warp_sums[warp]=odd;
        __syncthreads();
        if(!tid) {
            double x=0;
            for(int w=0;w<8;w++) x+=warp_sums[w];
            partial[tile*13+bit]=x;
        }
        __syncthreads();
    }
    if(!tid) partial[tile*13+12]=values[0];
}
extern "C" __global__ void hierarchical_marginal_finish(U nl, U tiles,
    int width, const double* partial, double* marginal) {
    U i=(U)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nl*(width+1)) return;
    U l=i/(width+1); int k=i%(width+1);
    double total=0;
    for(U t=0;t<tiles;t++) {
        if(k==width) total+=partial[(l*tiles+t)*13+12];
        else if(k<12) total+=partial[(l*tiles+t)*13+k];
        else if((t>>(k-12))&1) total+=partial[(l*tiles+t)*13+12];
    }
    marginal[i]=total;
}
extern "C" __global__ void coset_lookup_lut_stats(U nl, U nt, int chunks,
    const U* tables, const U* sizes, const U* offsets, const int* ranks,
    const double* sums, double* next, double* maxima, U* indices, double* totals) {
    U i=(U)blockIdx.x*blockDim.x+threadIdx.x, stride=(U)gridDim.x*blockDim.x;
    double maximum=-1,total=0; U index=~0ULL;
    for(;i<nl*nt;i+=stride) {
        U l=i%nl, t=i/nl;
        double x=0;
        if(sizes[l]) {
            U key=0;
            for(int c=0;c<chunks;c++) key^=tables[((U)c*256+((t>>(c*8))&255))*nl+l];
            double scale=__longlong_as_double(((U)(1023-ranks[l]))<<52);
            if(key<sizes[l]) x=sums[offsets[l]+key]*scale;
        }
        next[i]=x; total+=x;
        if(x>maximum || (x==maximum && i<index)) {maximum=x;index=i;}
    }
    combine_stats(maximum,index,total);
    __shared__ double wm[8],ws[8]; __shared__ U wi[8];
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    if(!lane) {wm[warp]=maximum;wi[warp]=index;ws[warp]=total;}
    __syncthreads();
    if(warp==0) {
        maximum=lane<8?wm[lane]:-1; index=lane<8?wi[lane]:~0ULL; total=lane<8?ws[lane]:0;
        combine_stats(maximum,index,total);
        if(!lane) {maxima[blockIdx.x]=maximum;indices[blockIdx.x]=index;totals[blockIdx.x]=total;}
    }
}
