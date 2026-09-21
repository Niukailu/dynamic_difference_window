// Rust host helpers. Probability arithmetic remains FP64.
extern "C" __global__ void window_values(U base, int width, const int *bits,
                                         U count, U *values) {
  U i = (U)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    U value = base;
    for (int j = 0; j < width; j++)
      value |= ((i >> j) & 1) << bits[j];
    values[i] = value;
  }
}
extern "C" __global__ void row_mass(const double *prob, U nr, double *mass) {
  U row = blockIdx.x;
  double sum = 0;
  for (U j = threadIdx.x; j < nr; j += 256)
    sum += prob[row * nr + j];
  __shared__ double warp[8];
  for (int d = 16; d; d >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, d);
  if (!(threadIdx.x & 31))
    warp[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (!threadIdx.x) {
    double total = 0;
    for (int j = 0; j < 8; j++)
      total += warp[j];
    mass[row] = total;
  }
}
extern "C" __global__ void scan_tiles(const U *input, U count, U *output,
                                      U *sums) {
  __shared__ U values[256];
  int tid = threadIdx.x;
  U i = (U)blockIdx.x * 256 + tid;
  values[tid] = i < count ? input[i] : 0;
  __syncthreads();
  for (int d = 1; d < 256; d *= 2) {
    U x = tid >= d ? values[tid - d] : 0;
    __syncthreads();
    values[tid] += x;
    __syncthreads();
  }
  if (i < count)
    output[i] = tid ? values[tid - 1] : 0;
  if (tid == 255)
    sums[blockIdx.x] = values[255];
}
extern "C" __global__ void scan_add(U *output, U count, const U *offsets) {
  U i = (U)blockIdx.x * 256 + threadIdx.x;
  if (i < count)
    output[i] += offsets[blockIdx.x];
}
extern "C" __global__ void scale_matrix(double *values, U count,
                                        double inverse) {
  for (U i = (U)blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += (U)gridDim.x * blockDim.x)
    values[i] *= inverse;
}
extern "C" __global__ void window_moments(const double *marginal, U nl,
                                          int width, const int *mapping,
                                          U right_base, const U *bases,
                                          const int *ranks, const U *support,
                                          double *output) {
  int bit = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  double sums[5] = {0, 0, 0, 0, 0};
  for (U row = threadIdx.x; row < nl; row += 256) {
    double mass = marginal[row * (width + 1) + width];
    double ones = mapping[bit] >= 0 ? marginal[row * (width + 1) + mapping[bit]]
                                    : ((right_base >> bit) & 1 ? mass : 0);
    if ((bases[row] >> bit) & 1)
      ones = mass - ones;
    double weight = __longlong_as_double((U)(1023 - ranks[row]) << 52);
    sums[0] += ones;
    sums[1] += ones * weight;
    if ((support[row] >> bit) & 1)
      sums[2] += mass * weight / max(__popcll(support[row]), 1);
    sums[3] += mass;
    sums[4] += mass * weight;
  }
  __shared__ double shared[5][8];
  for (int j = 0; j < 5; j++) {
    for (int d = 16; d; d >>= 1)
      sums[j] += __shfl_down_sync(0xffffffff, sums[j], d);
    if (!lane)
      shared[j][warp] = sums[j];
  }
  __syncthreads();
  if (!threadIdx.x)
    for (int j = 0; j < 5; j++) {
      double total = 0;
      for (int w = 0; w < 8; w++)
        total += shared[j][w];
      output[j * BITS + bit] = total;
    }
}
extern "C" __global__ void peer_unpack(const U *pointers, double *next, U count,
                                       U rows, U cols, int devices,
                                       int destination, double inverse) {
  for (U i = (U)blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += (U)gridDim.x * blockDim.x) {
    U row = i / (cols * devices), col = i % (cols * devices);
    const double *source = (const double *)pointers[col / cols];
    next[i] =
        source[((U)destination * rows + row) * cols + col % cols] * inverse;
  }
}
extern "C" __global__ void transpose_matrix(const double *source,
                                            double *target, U rows, U cols) {
  // One block covers a 32x32 tile using a linear grid, avoiding grid.y limits.
  __shared__ double tile[32][33];
  U tiles_x = (cols + 31) / 32;
  U bx = blockIdx.x % tiles_x, by = blockIdx.x / tiles_x;
  int x = threadIdx.x & 31, y = threadIdx.x >> 5;
  for (int j = 0; j < 32; j += 8)
    if (by * 32 + y + j < rows && bx * 32 + x < cols)
      tile[y + j][x] = source[(by * 32 + y + j) * cols + bx * 32 + x];
  __syncthreads();
  for (int j = 0; j < 32; j += 8)
    if (bx * 32 + y + j < cols && by * 32 + x < rows)
      target[(bx * 32 + y + j) * rows + by * 32 + x] = tile[x][y + j];
}
extern "C" __global__ void point_matrix(double *values, U count, U point) {
  for (U i = (U)blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += (U)gridDim.x * blockDim.x)
    values[i] = i == point ? 1 : 0;
}
extern "C" __global__ void dot_tiles(const double *a, const double *b, U count,
                                     double *result) {
  double value = 0;
  for (U i = (U)blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += (U)gridDim.x * blockDim.x)
    value += a[i] * b[i];
  __shared__ double sums[8];
  for (int d = 16; d; d >>= 1)
    value += __shfl_down_sync(0xffffffff, value, d);
  if (!(threadIdx.x & 31))
    sums[threadIdx.x >> 5] = value;
  __syncthreads();
  if (!threadIdx.x) {
    double total = 0;
    for (int j = 0; j < 8; j++)
      total += sums[j];
    result[blockIdx.x] = total;
  }
}

// Endpoint posterior over the changed scalar coordinate, without a product
// matrix.
extern "C" __global__ void posterior_rows(const double *forward,
                                          const double *backward, U cols,
                                          double *rows) {
  U row = blockIdx.x;
  double sum = 0;
  for (U col = threadIdx.x; col < cols; col += 256)
    sum += forward[row * cols + col] * backward[row * cols + col];
  __shared__ double warp[8];
  for (int d = 16; d; d >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, d);
  if (!(threadIdx.x & 31))
    warp[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (!threadIdx.x) {
    double total = 0;
    for (int w = 0; w < 8; w++)
      total += warp[w];
    rows[row] = total;
  }
}
extern "C" __global__ void posterior_halves(const double *rows, U count,
                                            double *result) {
  int bit = blockIdx.x >> 1, value = blockIdx.x & 1;
  double sum = 0;
  for (U row = threadIdx.x; row < count; row += 256)
    if (((row >> bit) & 1) == value)
      sum += rows[row];
  __shared__ double warp[8];
  for (int d = 16; d; d >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, d);
  if (!(threadIdx.x & 31))
    warp[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (!threadIdx.x) {
    double total = 0;
    for (int w = 0; w < 8; w++)
      total += warp[w];
    result[blockIdx.x] = total;
  }
}
