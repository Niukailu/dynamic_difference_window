// Exact virtual probability matrices: no dense nt*nl output allocation.
// view = [columns,chunks,tables,sizes,offsets,ranks,sums,inverse_bits].
__device__ __forceinline__ double implicit_value(const U *view, U row, U col) {
  U columns = view[0];
  const U *sizes = (const U *)view[3];
  if (!sizes[col])
    return 0;
  const U *tables = (const U *)view[2];
  U key = 0;
  for (int c = 0; c < (int)view[1]; c++)
    key ^= tables[((U)c * 256 + ((row >> (8 * c)) & 255)) * columns + col];
  if (key >= sizes[col])
    return 0;
  const U *offsets = (const U *)view[4];
  const int *ranks = (const int *)view[5];
  const double *sums = (const double *)view[6];
  return sums[offsets[col] + key] *
         __longlong_as_double((U)(1023 - ranks[col]) << 52) *
         __longlong_as_double(view[7]);
}
extern "C" __global__ void implicit_row_mass(const double *marginal, U width,
                                             double *value, U count) {
  U i = (U)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    value[i] = marginal[i * (width + 1) + width];
}
extern "C" __global__ void implicit_scalar(const U *view, U row, U col,
                                           double *value) {
  if (!threadIdx.x)
    *value = implicit_value(view, row, col);
}
extern "C" __global__ void implicit_materialize(const U *view, U rows, U cols,
                                                double *values) {
  for (U i = (U)blockIdx.x * blockDim.x + threadIdx.x; i < rows * cols;
       i += (U)gridDim.x * blockDim.x)
    values[i] = implicit_value(view, i / cols, i % cols);
}
extern "C" __global__ void
implicit_reduce(const U *view, U nr, const int *ranks, const int *outside,
                const U *reducers, const U *quotient_mask, const U *offsets,
                const U *sizes, double *sums) {
  U l = blockIdx.x;
  int h = ranks[l] - outside[l], low = h < 5 ? h : 5;
  int group = 1 << low, lane = threadIdx.x & (group - 1);
  U count = 1ULL << h, base = offsets[l];
  for (U bucket = threadIdx.x / group; bucket < sizes[l];
       bucket += 256 / group) {
    U coordinate = deposit(bucket, quotient_mask[l]);
    for (int j = 0; j < low; j++)
      if (lane & (1 << j))
        coordinate ^= reducers[l * 20 + j];
    double sum = 0;
    U iterations = count / group;
    for (U j = 0; j < iterations; j++) {
      sum += implicit_value(view, l, coordinate);
      if (j + 1 < iterations)
        coordinate ^= reducers[l * 20 + low + __ffsll(j + 1) - 1];
    }
    unsigned active = __activemask();
    for (int d = group / 2; d; d >>= 1)
      sum += __shfl_down_sync(active, sum, d, group);
    if (!lane)
      sums[base + bucket] = sum;
  }
}
// Cache row/bit marginals while collecting statistics: the next window
// selection reuses these values instead of scanning the entire virtual matrix
// again.
extern "C" __global__ void implicit_marginal_stats(const U *view, U nr,
                                                   int width, double *marginal,
                                                   double *maxima, U *indices,
                                                   double *totals) {
  U row = blockIdx.x;
  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  double upper[12] = {0}, total = 0, maximum = -1;
  U index = ~0ULL;
  for (U col = tid; col < nr; col += 256) {
    double x = implicit_value(view, row, col);
    total += x;
    if (x > maximum || (x == maximum && row * nr + col < index)) {
      maximum = x;
      index = row * nr + col;
    }
    if (x != 0)
      for (int k = 0; k < 12; k++)
        if ((col >> (k + 8)) & 1)
          upper[k] += x;
  }
  __shared__ double marginal_warps[21][8], wm[8], ws[8];
  __shared__ U wi[8];
  for (int k = 0; k <= width; k++) {
    double x = k == width ? total
               : k < 8    ? ((tid >> k) & 1 ? total : 0)
                          : upper[k - 8];
    for (int d = 16; d; d >>= 1)
      x += __shfl_down_sync(0xffffffff, x, d);
    if (!lane)
      marginal_warps[k][warp] = x;
  }
  __syncthreads();
  if (tid <= width) {
    double x = 0;
    for (int w = 0; w < 8; w++)
      x += marginal_warps[tid][w];
    marginal[row * (width + 1) + tid] = x;
  }
  combine_stats(maximum, index, total);
  if (!lane) {
    wm[warp] = maximum;
    wi[warp] = index;
    ws[warp] = total;
  }
  __syncthreads();
  if (!warp) {
    maximum = lane < 8 ? wm[lane] : -1;
    index = lane < 8 ? wi[lane] : ~0ULL;
    total = lane < 8 ? ws[lane] : 0;
    combine_stats(maximum, index, total);
    if (!lane) {
      maxima[row] = maximum;
      indices[row] = index;
      totals[row] = total;
    }
  }
}
