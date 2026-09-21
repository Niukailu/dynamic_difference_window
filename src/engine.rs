use crate::{
    args,
    cuda::{Buffer, Context, Module},
    model::{Config, Window},
};
use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, sync::Arc};
pub const KERNELS: &str = include_str!("../cuda/kernels.cu");
pub const IMPLICIT: &str = include_str!("../cuda/implicit.cu");
pub const HELPERS: &str = include_str!("../cuda/runtime.cu");
pub fn kernel_hash() -> String {
    format!(
        "{:x}",
        Sha256::digest(format!("{KERNELS}\n{HELPERS}\n{IMPLICIT}").as_bytes())
    )
}
#[derive(Debug)]
pub struct EmptyDistribution;
impl std::fmt::Display for EmptyDistribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "implicit transition retained no paths")
    }
}
impl std::error::Error for EmptyDistribution {}
/// Lossless representation of an FP64 matrix by its preceding transition's cosets.
pub struct Compressed {
    pub rows: usize,
    pub cols: usize,
    view: Buffer,
    owned: Vec<Buffer>,
    descriptor: Vec<u64>,
    marginal: Buffer,
}
impl Compressed {
    pub fn bytes(&self) -> usize {
        self.view.bytes + self.marginal.bytes + self.owned.iter().map(|b| b.bytes).sum::<usize>()
    }
    pub fn buckets(&self) -> usize {
        self.owned.last().unwrap().bytes / 8
    }
    pub fn normalization(&self) -> f64 {
        f64::from_bits(self.descriptor[7])
    }
}
pub struct Matrix {
    pub buffer: Buffer,
    pub rows: usize,
    pub cols: usize,
}
impl Matrix {
    pub fn size(&self) -> usize {
        self.rows * self.cols
    }
    pub fn ptr(&self) -> u64 {
        self.buffer.ptr
    }
    pub fn value(&self, row: usize, col: usize) -> Result<f64> {
        ensure!(
            row < self.rows && col < self.cols,
            "matrix index out of bounds"
        );
        self.buffer.scalar_f64(row * self.cols + col)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub peak: f64,
    pub total: f64,
    pub index: usize,
}
pub struct Engine {
    pub context: Arc<Context>,
    pub config: Config,
    module: Module,
    workspace: HashMap<String, Buffer>,
}
fn blocks(n: usize, t: usize) -> usize {
    n.div_ceil(t)
}
impl Engine {
    pub fn new(config: &Config, device: i32, memory_gib: f64) -> Result<Self> {
        config.validate()?;
        let context = Context::new(device, memory_gib)?;
        let n = config.word_bits;
        let mask = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
        let (a, b, c) = if config.cipher == "simon" {
            (8, 1, 2)
        } else {
            (5, 0, 1)
        };
        let source=format!("#define BITS {n}\n#define WORD_MASK {mask}ULL\n#define RA {a}\n#define RB {b}\n#define RC {c}\n#define LINEAR_MODE {}\n{KERNELS}\n{HELPERS}\n{IMPLICIT}",u8::from(config.mode=="linear"));
        let module = Module::new(context.clone(), &source)?;
        Ok(Self {
            context,
            config: config.clone(),
            module,
            workspace: HashMap::new(),
        })
    }
    fn reserve(&mut self, name: &str, bytes: usize) -> Result<u64> {
        if self.workspace.get(name).is_none_or(|b| b.bytes < bytes) {
            self.workspace.remove(name);
            let b = Buffer::new(self.context.clone(), bytes)?;
            self.workspace.insert(name.into(), b);
        }
        Ok(self.workspace[name].ptr)
    }
    fn upload<T: crate::cuda::Pod>(&mut self, name: &str, values: &[T]) -> Result<u64> {
        let ptr = self.reserve(name, std::mem::size_of_val(values))?;
        self.workspace[name].upload(values)?;
        Ok(ptr)
    }
    pub fn allocate(&self, rows: usize, cols: usize) -> Result<Matrix> {
        let bytes = rows
            .checked_mul(cols)
            .and_then(|v| v.checked_mul(8))
            .ok_or_else(|| anyhow::anyhow!("matrix size overflow"))?;
        Ok(Matrix {
            buffer: Buffer::new(self.context.clone(), bytes)?,
            rows,
            cols,
        })
    }
    pub fn point(&mut self, rows: usize, cols: usize, index: usize) -> Result<Matrix> {
        ensure!(index < rows * cols, "point outside matrix");
        let result = self.allocate(rows, cols)?;
        self.module.launch(
            "point_matrix",
            blocks(result.size(), 256).min(65535),
            256,
            args![result.ptr(), result.size(), index],
        )?;
        Ok(result)
    }
    pub fn from_host(&self, rows: usize, cols: usize, values: &[f64]) -> Result<Matrix> {
        ensure!(rows * cols == values.len(), "host matrix shape mismatch");
        let result = self.allocate(rows, cols)?;
        result.buffer.upload(values)?;
        Ok(result)
    }
    pub fn scale(&mut self, matrix: &Matrix, inverse: f64) -> Result<()> {
        ensure!(inverse.is_finite() && inverse > 0., "invalid normalization");
        self.module.launch(
            "scale_matrix",
            blocks(matrix.size(), 256).min(65535),
            256,
            args![matrix.ptr(), matrix.size(), inverse],
        )
    }
    fn basis(&mut self, left: &Window) -> Result<(u64, u64, u64, u64)> {
        let nl = left.size();
        let bits = self.upload("left_bits", &left.bits)?;
        let values = self.reserve("left_values", nl * 8)?;
        let bases = self.reserve("bases", nl * 8)?;
        let vectors = self.reserve("vectors", nl * self.config.word_bits as usize * 8)?;
        let ranks = self.reserve("ranks", nl * 4)?;
        let support = self.reserve("support", nl * 8)?;
        self.module.launch(
            "window_values",
            blocks(nl, 128),
            128,
            args![left.base, left.bits.len() as i32, bits, nl, values],
        )?;
        self.module.launch(
            "basis",
            blocks(nl, 128),
            128,
            args![values, nl, bases, vectors, ranks, support],
        )?;
        Ok((bases, vectors, ranks, support))
    }
    fn prefix(&mut self, input: u64, count: usize, output: u64, level: usize) -> Result<()> {
        let tiles = blocks(count, 256);
        let sums = self.reserve(&format!("scan_sums_{level}"), tiles * 8)?;
        self.module
            .launch("scan_tiles", tiles, 256, args![input, count, output, sums])?;
        if tiles > 1 {
            let offsets = self.reserve(&format!("scan_offsets_{level}"), tiles * 8)?;
            self.prefix(sums, tiles, offsets, level + 1)?;
            self.module
                .launch("scan_add", tiles, 256, args![output, count, offsets])?;
        }
        Ok(())
    }
    fn finish(&mut self, count: usize, maxima: u64, indices: u64, totals: u64) -> Result<Stats> {
        let result = self.reserve("stats", 24)?;
        self.module.launch(
            "finish_stats",
            1,
            256,
            args![count, maxima, indices, totals, result],
        )?;
        let values = self.workspace["stats"].download::<f64>(3)?;
        ensure!(
            values[0].is_finite() && values[1].is_finite(),
            "invalid distribution statistics"
        );
        Ok(Stats {
            peak: values[0],
            total: values[1],
            index: values[2] as usize,
        })
    }
    pub fn summary(&mut self, pointer: u64, count: usize) -> Result<Stats> {
        let groups = blocks(count, 256).min(8192);
        let maxima = self.reserve("maxima", groups * 8)?;
        let indices = self.reserve("indices", groups * 8)?;
        let totals = self.reserve("totals", groups * 8)?;
        self.module.launch(
            "distribution_stats",
            groups,
            256,
            args![pointer, count, maxima, indices, totals],
        )?;
        self.finish(groups, maxima, indices, totals)
    }
    pub fn step(
        &mut self,
        prob: &Matrix,
        left: &Window,
        right: &Window,
        target: &Window,
        spare: Option<Matrix>,
    ) -> Result<(Matrix, Stats)> {
        ensure!(
            (prob.rows, prob.cols) == (left.size(), right.size()),
            "probability shape does not match windows"
        );
        let nl = prob.rows;
        let nr = prob.cols;
        let nt = target.size();
        let n = self.config.word_bits as usize;
        let (bases, vectors, ranks, _) = self.basis(left)?;
        let outside = self.reserve("outside", nl * 4)?;
        let packed = self.reserve("packed", nl * n * 8)?;
        self.module.launch(
            "restrict_basis",
            blocks(nl, 128),
            128,
            args![nl, right.mask(), bases, vectors, ranks, outside],
        )?;
        self.module.launch(
            "pack_basis",
            blocks(nl * n, 128).min(65535),
            128,
            args![nl, right.mask(), vectors, packed],
        )?;
        let mass = self.reserve("row_mass", nl * 8)?;
        self.module
            .launch("row_mass", nl, 256, args![prob.ptr(), nr, mass])?;
        let reducers = self.reserve("reducers", nl * 20 * 8)?;
        let masks = self.reserve("masks", nl * 8)?;
        let sizes = self.reserve("sizes", nl * 8)?;
        let offsets = self.reserve("offsets", nl * 8)?;
        self.module.launch(
            "coset_plan",
            blocks(nl, 128),
            128,
            args![nl, nr, packed, ranks, outside, mass, reducers, masks, sizes],
        )?;
        self.prefix(sizes, nl, offsets, 0)?;
        // Download two small planning arrays only at their last elements.
        let last_offset = self.read_u64(offsets + (nl as u64 - 1) * 8)?;
        let last_size = self.read_u64(sizes + (nl as u64 - 1) * 8)?;
        let buckets = (last_offset + last_size) as usize;
        let sums = self.reserve("coset_sums", buckets * 8)?;
        let output = match spare {
            Some(m) if (m.rows, m.cols) == (nt, nl) => m,
            other => {
                drop(other);
                self.allocate(nt, nl)?
            }
        };
        self.module.launch(
            "coset_reduce",
            nl,
            256,
            args![
                prob.ptr(),
                nr,
                ranks,
                outside,
                reducers,
                masks,
                offsets,
                sizes,
                sums
            ],
        )?;
        let width = target.bits.len() as i32;
        let chunks = ((width + 7) / 8).max(1);
        let bits = self.upload("target_bits", &target.bits)?;
        let columns = self.reserve("columns", (width as usize + 1) * nl * 8)?;
        let tables = self.reserve("tables", chunks as usize * 256 * nl * 8)?;
        self.module.launch(
            "affine_columns",
            blocks(nl * (width as usize + 1), 128),
            128,
            args![
                nl,
                width,
                bits,
                target.base,
                right.mask(),
                right.base,
                bases,
                vectors,
                ranks,
                outside,
                reducers,
                masks,
                columns
            ],
        )?;
        self.module.launch(
            "affine_tables",
            blocks(nl * 256 * chunks as usize, 128).min(65535),
            128,
            args![nl, width, chunks, columns, tables],
        )?;
        let groups = blocks(output.size(), 256).min(8192);
        let maxima = self.reserve("maxima", groups * 8)?;
        let indices = self.reserve("indices", groups * 8)?;
        let totals = self.reserve("totals", groups * 8)?;
        self.module.launch(
            "coset_lookup_lut_stats",
            groups,
            256,
            args![
                nl,
                nt,
                chunks,
                tables,
                sizes,
                offsets,
                ranks,
                sums,
                output.ptr(),
                maxima,
                indices,
                totals
            ],
        )?;
        Ok((output, self.finish(groups, maxima, indices, totals)?))
    }
    fn read_u64(&self, pointer: u64) -> Result<u64> {
        crate::cuda::read_u64(&self.context, pointer)
    }
    pub fn moments(&mut self, prob: &Matrix, left: &Window, right: &Window) -> Result<Vec<f64>> {
        let nl = prob.rows;
        let nr = prob.cols;
        let width = right.bits.len();
        let n = self.config.word_bits as usize;
        let (bases, _, ranks, support) = self.basis(left)?;
        let tiles = nr.div_ceil(4096);
        let partial = self.reserve("marginal_partial", nl * tiles * 13 * 8)?;
        let marginal = self.reserve("marginal", nl * (width + 1) * 8)?;
        self.module.launch(
            "hierarchical_marginal_tiles",
            nl * tiles,
            256,
            args![prob.ptr(), nr, tiles, partial],
        )?;
        self.module.launch(
            "hierarchical_marginal_finish",
            blocks(nl * (width + 1), 128),
            128,
            args![nl, tiles, width as i32, partial, marginal],
        )?;
        let mut mapping = vec![-1i32; n];
        for (i, &bit) in right.bits.iter().enumerate() {
            mapping[bit as usize] = i as i32;
        }
        let mapping = self.upload("mapping", &mapping)?;
        let output = self.reserve("moments", 5 * n * 8)?;
        self.module.launch(
            "window_moments",
            n,
            256,
            args![
                marginal,
                nl,
                width as i32,
                mapping,
                right.base,
                bases,
                ranks,
                support,
                output
            ],
        )?;
        self.workspace["moments"].download(5 * n)
    }
    pub fn candidates(
        &mut self,
        prob: &Matrix,
        left: &Window,
        right: &Window,
        width: usize,
        include: Option<&Window>,
        count: usize,
    ) -> Result<Vec<Window>> {
        let moments = self.moments(prob, left, right)?;
        select(
            &moments,
            self.config.word_bits as usize,
            width,
            include,
            count,
        )
    }
    pub fn retained_mass(
        &mut self,
        prob: &Matrix,
        left: &Window,
        right: &Window,
        target: &Window,
    ) -> Result<f64> {
        let nl = prob.rows;
        let nr = prob.cols;
        let (bases, vectors, ranks, _) = self.basis(left)?;
        let outside = self.reserve("outside", nl * 4)?;
        self.module.launch(
            "restrict_basis",
            blocks(nl, 128),
            128,
            args![nl, target.mask(), bases, vectors, ranks, outside],
        )?;
        let width = right.bits.len() as i32;
        let chunks = ((width + 7) / 8).max(1);
        let bits = self.upload("target_bits", &right.bits)?;
        let columns = self.reserve("columns", (width as usize + 1) * nl * 8)?;
        let tables = self.reserve("tables", chunks as usize * 256 * nl * 8)?;
        self.module.launch(
            "mass_columns",
            blocks(nl * (width as usize + 1), 128),
            128,
            args![
                nl,
                width,
                bits,
                target.mask(),
                target.base,
                right.base,
                bases,
                vectors,
                outside,
                columns
            ],
        )?;
        self.module.launch(
            "affine_tables",
            blocks(nl * 256 * chunks as usize, 128).min(65535),
            128,
            args![nl, width, chunks, columns, tables],
        )?;
        let tiles = nr.div_ceil(4096);
        let partial = self.reserve("mass_partial", nl * tiles * 8)?;
        self.module.launch(
            "retained_mass_tiles",
            nl * tiles,
            256,
            args![prob.ptr(), nl, nr, tiles, chunks, tables, outside, partial],
        )?;
        Ok(self.summary(partial, nl * tiles)?.total)
    }
    pub fn transpose(&mut self, value: &Matrix) -> Result<Matrix> {
        let result = self.allocate(value.cols, value.rows)?;
        self.module.launch(
            "transpose_matrix",
            value.rows.div_ceil(32) * value.cols.div_ceil(32),
            256,
            args![value.ptr(), result.ptr(), value.rows, value.cols],
        )?;
        Ok(result)
    }
    pub fn adjoint(
        &mut self,
        value: &Matrix,
        left: &Window,
        right: &Window,
        target: &Window,
    ) -> Result<Matrix> {
        let transposed = self.transpose(value)?;
        let (result, _) = self.step(&transposed, left, target, right, None)?;
        self.transpose(&result)
    }
    pub fn dot(&mut self, a: &Matrix, b: &Matrix) -> Result<f64> {
        ensure!((a.rows, a.cols) == (b.rows, b.cols), "dot shape mismatch");
        let groups = blocks(a.size(), 256).min(8192);
        let partial = self.reserve("dot_partial", groups * 8)?;
        self.module.launch(
            "dot_tiles",
            groups,
            256,
            args![a.ptr(), b.ptr(), a.size(), partial],
        )?;
        Ok(self.summary(partial, groups)?.total)
    }
    pub fn compressed_point(&self) -> Result<Compressed> {
        let mut owned = vec![];
        let mut make = |values: &[u64]| -> Result<u64> {
            let b = Buffer::new(self.context.clone(), values.len() * 8)?;
            b.upload(values)?;
            let p = b.ptr;
            owned.push(b);
            Ok(p)
        };
        let tables = make(&[0u64; 256])?;
        let sizes = make(&[1])?;
        let offsets = make(&[0])?;
        let ranks = make(&[0])?;
        let sums = make(&[1f64.to_bits()])?;
        let descriptor = vec![1, 1, tables, sizes, offsets, ranks, sums, 1f64.to_bits()];
        let view = Buffer::new(self.context.clone(), 64)?;
        view.upload(&descriptor)?;
        let marginal = Buffer::new(self.context.clone(), 8)?;
        marginal.upload(&[1f64])?;
        Ok(Compressed {
            marginal,
            rows: 1,
            cols: 1,
            view,
            owned,
            descriptor,
        })
    }
    pub fn compressed_step(
        &mut self,
        prob: &Compressed,
        left: &Window,
        right: &Window,
        target: &Window,
    ) -> Result<(Compressed, Stats)> {
        ensure!(
            (prob.rows, prob.cols) == (left.size(), right.size()),
            "implicit shape mismatch"
        );
        let nl = left.size();
        let nr = right.size();
        let nt = target.size();
        let n = self.config.word_bits as usize;
        let (bases, vectors, ranks, _) = self.basis(left)?;
        let outside = self.reserve("outside", nl * 4)?;
        let packed = self.reserve("packed", nl * n * 8)?;
        self.module.launch(
            "restrict_basis",
            blocks(nl, 128),
            128,
            args![nl, right.mask(), bases, vectors, ranks, outside],
        )?;
        self.module.launch(
            "pack_basis",
            blocks(nl * n, 128).min(65535),
            128,
            args![nl, right.mask(), vectors, packed],
        )?;
        let mass = self.reserve("row_mass", nl * 8)?;
        // A zero row may allocate empty buckets; it never changes numerical results.
        self.module
            .launch("fill_one", blocks(nl, 128), 128, args![mass, nl])?;
        let reducers = self.reserve("reducers", nl * 20 * 8)?;
        let masks = self.reserve("masks", nl * 8)?;
        let sizes = self.reserve("sizes", nl * 8)?;
        let offsets = self.reserve("offsets", nl * 8)?;
        self.module.launch(
            "coset_plan",
            blocks(nl, 128),
            128,
            args![nl, nr, packed, ranks, outside, mass, reducers, masks, sizes],
        )?;
        self.prefix(sizes, nl, offsets, 0)?;
        let buckets = (self.read_u64(offsets + (nl as u64 - 1) * 8)?
            + self.read_u64(sizes + (nl as u64 - 1) * 8)?) as usize;
        let sums = self.reserve("coset_sums", buckets * 8)?;
        self.module.launch(
            "implicit_reduce",
            nl,
            256,
            args![
                prob.view.ptr,
                nr,
                ranks,
                outside,
                reducers,
                masks,
                offsets,
                sizes,
                sums
            ],
        )?;
        let width = target.bits.len() as i32;
        let chunks = ((width + 7) / 8).max(1);
        let bits = self.upload("target_bits", &target.bits)?;
        let columns = self.reserve("columns", (width as usize + 1) * nl * 8)?;
        let tables = self.reserve("tables", chunks as usize * 256 * nl * 8)?;
        self.module.launch(
            "affine_columns",
            blocks(nl * (width as usize + 1), 128),
            128,
            args![
                nl,
                width,
                bits,
                target.base,
                right.mask(),
                right.base,
                bases,
                vectors,
                ranks,
                outside,
                reducers,
                masks,
                columns
            ],
        )?;
        self.module.launch(
            "affine_tables",
            blocks(nl * 256 * chunks as usize, 128).min(65535),
            128,
            args![nl, width, chunks, columns, tables],
        )?;
        let owned = ["tables", "sizes", "offsets", "ranks", "coset_sums"]
            .iter()
            .map(|key| self.workspace.remove(*key).unwrap())
            .collect();
        let mut descriptor = vec![
            nl as u64,
            chunks as u64,
            tables,
            sizes,
            offsets,
            ranks,
            sums,
            1f64.to_bits(),
        ];
        let view = Buffer::new(self.context.clone(), 64)?;
        view.upload(&descriptor)?;
        let row_width = nl.trailing_zeros() as usize;
        let marginal = Buffer::new(self.context.clone(), nt * (row_width + 1) * 8)?;
        let groups = nt;
        let maxima = self.reserve("maxima", groups * 8)?;
        let indices = self.reserve("indices", groups * 8)?;
        let totals = self.reserve("totals", groups * 8)?;
        self.module.launch(
            "implicit_marginal_stats",
            nt,
            256,
            args![
                view.ptr,
                nl,
                row_width as i32,
                marginal.ptr,
                maxima,
                indices,
                totals
            ],
        )?;
        let stats = self.finish(groups, maxima, indices, totals)?;
        if stats.peak <= 0. {
            return Err(EmptyDistribution.into());
        }
        descriptor[7] = (1. / stats.peak).to_bits();
        view.upload(&descriptor)?;
        Ok((
            Compressed {
                marginal,
                rows: nt,
                cols: nl,
                view,
                owned,
                descriptor,
            },
            stats,
        ))
    }
    pub fn compressed_value(&mut self, prob: &Compressed, row: usize, col: usize) -> Result<f64> {
        ensure!(
            row < prob.rows && col < prob.cols,
            "implicit index out of bounds"
        );
        let value = self.reserve("implicit_value", 8)?;
        self.module.launch(
            "implicit_scalar",
            1,
            1,
            args![prob.view.ptr, row, col, value],
        )?;
        self.workspace["implicit_value"].scalar_f64(0)
    }
    pub fn compressed_materialize(&mut self, prob: &Compressed) -> Result<Matrix> {
        let matrix = self.allocate(prob.rows, prob.cols)?;
        self.module.launch(
            "implicit_materialize",
            blocks(matrix.size(), 256).min(65535),
            256,
            args![prob.view.ptr, prob.rows, prob.cols, matrix.ptr()],
        )?;
        Ok(matrix)
    }
    pub fn compressed_candidates(
        &mut self,
        prob: &Compressed,
        left: &Window,
        right: &Window,
        width: usize,
        include: Option<&Window>,
    ) -> Result<Vec<Window>> {
        let nl = left.size();
        let rw = right.bits.len();
        let n = self.config.word_bits as usize;
        let (bases, _, ranks, support) = self.basis(left)?;
        let marginal = prob.marginal.ptr;
        let mut mapping = vec![-1i32; n];
        for (i, &bit) in right.bits.iter().enumerate() {
            mapping[bit as usize] = i as i32;
        }
        let mapping = self.upload("mapping", &mapping)?;
        let output = self.reserve("moments", 5 * n * 8)?;
        self.module.launch(
            "window_moments",
            n,
            256,
            args![marginal, nl, rw as i32, mapping, right.base, bases, ranks, support, output],
        )?;
        let moments = self.workspace["moments"].download::<f64>(5 * n)?;
        select(&moments, n, width, include, 1)
    }
    /// For each packed row bit, return endpoint mass in its zero and one halves.
    /// Both halves are summed directly to avoid cancellation for rare endpoint paths.
    pub fn posterior_halves(&mut self, forward: &Matrix, backward: &Matrix) -> Result<Vec<f64>> {
        ensure!(
            (forward.rows, forward.cols) == (backward.rows, backward.cols)
                && forward.rows.is_power_of_two()
                && forward.rows > 1,
            "invalid posterior shape"
        );
        let width = forward.rows.trailing_zeros() as usize;
        let rows = self.reserve("posterior_rows", forward.rows * 8)?;
        let halves = self.reserve("posterior_halves", width * 2 * 8)?;
        self.module.launch(
            "posterior_rows",
            forward.rows,
            256,
            args![forward.ptr(), backward.ptr(), forward.cols, rows],
        )?;
        self.module.launch(
            "posterior_halves",
            width * 2,
            256,
            args![rows, forward.rows, halves],
        )?;
        self.workspace["posterior_halves"].download(width * 2)
    }
    pub fn unpack(
        &mut self,
        pointers: &[u64],
        output: &Matrix,
        rows: usize,
        cols: usize,
        rank: usize,
        inverse: f64,
    ) -> Result<()> {
        let device_pointers = self.upload("peer_pointers", pointers)?;
        self.module.launch(
            "peer_unpack",
            blocks(output.size(), 256).min(65535),
            256,
            args![
                device_pointers,
                output.ptr(),
                output.size(),
                rows,
                cols,
                pointers.len() as i32,
                rank as i32,
                inverse
            ],
        )?;
        self.context.sync()
    }
}
pub fn select(
    moments: &[f64],
    n: usize,
    width: usize,
    include: Option<&Window>,
    count: usize,
) -> Result<Vec<Window>> {
    ensure!(
        width <= 20 && width <= n && count > 0,
        "invalid window search parameters"
    );
    ensure!(
        include.is_none_or(|w| w.bits.len() <= width),
        "included window exceeds width"
    );
    let mut base = 0u64;
    let mut scores = vec![0.; n];
    for bit in 0..n {
        let chosen = moments[bit] > moments[3 * n + bit] * 0.5;
        if chosen {
            base |= 1u64 << bit;
        }
        scores[bit] = ((if chosen {
            moments[4 * n + bit] - moments[n + bit]
        } else {
            moments[n + bit]
        }) + moments[2 * n + bit])
            .max(0.);
    }
    let mut ranked: Vec<_> = (0..n).collect();
    ranked.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let mandatory = include.map(|w| w.bits.clone()).unwrap_or_default();
    let mut selected = mandatory.clone();
    for &bit in &ranked {
        if selected.len() >= width {
            break;
        }
        if scores[bit] > 0. && !selected.contains(&(bit as i32)) {
            selected.push(bit as i32);
        }
    }
    selected.sort();
    if let Some(w) = include {
        base = w.base;
    }
    let mut result = vec![Window::new(base, selected.clone())?];
    let mut exchanges = vec![];
    for &old in &selected {
        if mandatory.contains(&old) {
            continue;
        }
        for new in 0..n {
            if !selected.contains(&(new as i32)) && scores[new] > 0. {
                exchanges.push((scores[old as usize] - scores[new], old, new as i32));
            }
        }
    }
    exchanges.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    for &(_, old, new) in exchanges.iter().take(count - 1) {
        let mut bits = selected.clone();
        bits.retain(|&b| b != old);
        bits.push(new);
        bits.sort();
        result.push(Window::new(base, bits)?);
    }
    Ok(result)
}
