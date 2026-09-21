//! Minimal CUDA driver boundary. Device memory never escapes as a Rust reference.
use anyhow::{bail, ensure, Result};
use std::{
    collections::HashMap,
    ffi::{c_char, c_int, c_void, CStr, CString},
    ptr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};
type Handle = *mut c_void;
extern "C" {
    fn cuInit(flags: u32) -> c_int;
    fn cuDeviceGet(device: *mut c_int, ordinal: c_int) -> c_int;
    fn cuDeviceGetAttribute(value: *mut c_int, attribute: c_int, device: c_int) -> c_int;
    fn cuDeviceGetName(name: *mut c_char, length: c_int, device: c_int) -> c_int;
    fn cuDevicePrimaryCtxRetain(context: *mut Handle, device: c_int) -> c_int;
    fn cuDevicePrimaryCtxRelease_v2(device: c_int) -> c_int;
    fn cuCtxSetCurrent(context: Handle) -> c_int;
    fn cuCtxSynchronize() -> c_int;
    fn cuCtxEnablePeerAccess(peer: Handle, flags: u32) -> c_int;
    fn cuMemAlloc_v2(pointer: *mut u64, bytes: usize) -> c_int;
    fn cuMemFree_v2(pointer: u64) -> c_int;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> c_int;
    fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> c_int;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> c_int;
    fn cuMemcpyPeer(dst: u64, dst_ctx: Handle, src: u64, src_ctx: Handle, bytes: usize) -> c_int;
    fn cuModuleLoadData(module: *mut Handle, image: *const c_void) -> c_int;
    fn cuModuleUnload(module: Handle) -> c_int;
    fn cuModuleGetFunction(function: *mut Handle, module: Handle, name: *const c_char) -> c_int;
    fn cuLaunchKernel(
        function: Handle,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        shared: u32,
        stream: Handle,
        args: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> c_int;
    fn cuGetErrorString(status: c_int, message: *mut *const c_char) -> c_int;
    fn nvrtcCreateProgram(
        program: *mut Handle,
        source: *const c_char,
        name: *const c_char,
        count: c_int,
        headers: *const *const c_char,
        names: *const *const c_char,
    ) -> c_int;
    fn nvrtcCompileProgram(program: Handle, count: c_int, options: *const *const c_char) -> c_int;
    fn nvrtcGetProgramLogSize(program: Handle, size: *mut usize) -> c_int;
    fn nvrtcGetProgramLog(program: Handle, log: *mut c_char) -> c_int;
    fn nvrtcGetPTXSize(program: Handle, size: *mut usize) -> c_int;
    fn nvrtcGetPTX(program: Handle, ptx: *mut c_char) -> c_int;
    fn nvrtcDestroyProgram(program: *mut Handle) -> c_int;
}
fn check(code: c_int) -> Result<()> {
    if code != 0 {
        unsafe {
            let mut message = ptr::null();
            cuGetErrorString(code, &mut message);
            bail!(
                "CUDA {code}: {}",
                if message.is_null() {
                    "unknown error".into()
                } else {
                    CStr::from_ptr(message).to_string_lossy()
                }
            );
        }
    }
    Ok(())
}
#[derive(Debug)]
pub struct Context {
    raw: usize,
    device: c_int,
    pub ordinal: i32,
    pub name: String,
    pub arch: String,
    limit: usize,
    allocated: AtomicUsize,
}
impl Context {
    pub fn new(ordinal: i32, memory_gib: f64) -> Result<Arc<Self>> {
        ensure!(
            memory_gib.is_finite() && memory_gib > 0.,
            "positive finite memory budget required"
        );
        unsafe {
            check(cuInit(0))?;
            let mut device = 0;
            check(cuDeviceGet(&mut device, ordinal))?;
            let mut major = 0;
            let mut minor = 0;
            check(cuDeviceGetAttribute(&mut major, 75, device))?;
            check(cuDeviceGetAttribute(&mut minor, 76, device))?;
            let mut name = [0i8; 256];
            check(cuDeviceGetName(name.as_mut_ptr(), 256, device))?;
            let mut raw = ptr::null_mut();
            check(cuDevicePrimaryCtxRetain(&mut raw, device))?;
            check(cuCtxSetCurrent(raw))?;
            Ok(Arc::new(Self {
                raw: raw as usize,
                device,
                ordinal,
                name: CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned(),
                arch: format!("compute_{major}{minor}"),
                limit: (memory_gib * 1073741824.) as usize,
                allocated: AtomicUsize::new(0),
            }))
        }
    }
    pub fn activate(&self) -> Result<()> {
        unsafe { check(cuCtxSetCurrent(self.raw as Handle)) }
    }
    pub fn sync(&self) -> Result<()> {
        self.activate()?;
        unsafe { check(cuCtxSynchronize()) }
    }
    pub fn peer(&self, other: &Self) -> Result<()> {
        self.activate()?;
        if self.device == other.device {
            return Ok(());
        }
        let status = unsafe { cuCtxEnablePeerAccess(other.raw as Handle, 0) };
        if status == 704 {
            Ok(())
        } else {
            check(status)
        }
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            cuDevicePrimaryCtxRelease_v2(self.device);
        }
    }
}
// Context handles may be activated on different threads. Exclusive engine access
// and owned Buffer values prevent concurrent mutation of the same allocation.
pub struct Buffer {
    pub ptr: u64,
    pub bytes: usize,
    pub context: Arc<Context>,
}
impl Buffer {
    pub fn new(context: Arc<Context>, bytes: usize) -> Result<Self> {
        context.activate()?;
        let bytes = bytes.max(8);
        let used = context.allocated.fetch_add(bytes, Ordering::SeqCst);
        if used.checked_add(bytes).is_none_or(|v| v > context.limit) {
            context.allocated.fetch_sub(bytes, Ordering::SeqCst);
            bail!(
                "device {} memory budget exceeded: {:.3} GiB requested",
                context.ordinal,
                used.saturating_add(bytes) as f64 / 1073741824.
            );
        }
        let mut free = 0;
        let mut total = 0;
        let mut pointer = 0;
        let result = unsafe {
            check(cuMemGetInfo_v2(&mut free, &mut total)).and_then(|_| {
                ensure!(
                    bytes as f64 <= free as f64 * 0.9,
                    "insufficient free GPU memory with 10% headroom"
                );
                check(cuMemAlloc_v2(&mut pointer, bytes))
            })
        };
        if let Err(error) = result {
            context.allocated.fetch_sub(bytes, Ordering::SeqCst);
            return Err(error);
        }
        Ok(Self {
            ptr: pointer,
            bytes,
            context,
        })
    }
    pub fn upload<T: Pod>(&self, values: &[T]) -> Result<()> {
        ensure!(
            std::mem::size_of_val(values) <= self.bytes,
            "upload out of bounds"
        );
        self.context.activate()?;
        unsafe {
            check(cuMemcpyHtoD_v2(
                self.ptr,
                values.as_ptr().cast(),
                std::mem::size_of_val(values),
            ))
        }
    }
    pub fn download<T: Pod>(&self, count: usize) -> Result<Vec<T>> {
        ensure!(
            count
                .checked_mul(size_of::<T>())
                .is_some_and(|bytes| bytes <= self.bytes),
            "download out of bounds"
        );
        self.context.activate()?;
        let mut values = vec![T::default(); count];
        unsafe {
            check(cuMemcpyDtoH_v2(
                values.as_mut_ptr().cast(),
                self.ptr,
                count * size_of::<T>(),
            ))?;
        }
        Ok(values)
    }
    pub fn scalar_f64(&self, index: usize) -> Result<f64> {
        ensure!(
            index
                .checked_add(1)
                .and_then(|n| n.checked_mul(8))
                .is_some_and(|bytes| bytes <= self.bytes),
            "scalar out of bounds"
        );
        self.context.activate()?;
        let mut value = 0.;
        unsafe {
            check(cuMemcpyDtoH_v2(
                (&mut value as *mut f64).cast(),
                self.ptr + (index * 8) as u64,
                8,
            ))?;
        }
        Ok(value)
    }
    pub fn copy_from(&self, source: &Self, offset: usize, bytes: usize) -> Result<()> {
        ensure!(
            bytes <= self.bytes
                && offset
                    .checked_add(bytes)
                    .is_some_and(|end| end <= source.bytes),
            "peer copy out of bounds"
        );
        self.context.activate()?;
        unsafe {
            check(cuMemcpyPeer(
                self.ptr,
                self.context.raw as Handle,
                source.ptr + offset as u64,
                source.context.raw as Handle,
                bytes,
            ))
        }
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        let _ = self.context.activate();
        unsafe {
            cuMemFree_v2(self.ptr);
        }
        self.context
            .allocated
            .fetch_sub(self.bytes, Ordering::SeqCst);
    }
}
mod sealed {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for u64 {}
    impl Sealed for i32 {}
}
pub trait Pod: sealed::Sealed + Copy + Default {}
impl Pod for f64 {}
impl Pod for u64 {}
impl Pod for i32 {}
pub enum Arg {
    U(u64),
    I(i32),
    F(f64),
}
impl From<u64> for Arg {
    fn from(v: u64) -> Self {
        Self::U(v)
    }
}
impl From<usize> for Arg {
    fn from(v: usize) -> Self {
        Self::U(v as u64)
    }
}
impl From<i32> for Arg {
    fn from(v: i32) -> Self {
        Self::I(v)
    }
}
impl From<f64> for Arg {
    fn from(v: f64) -> Self {
        Self::F(v)
    }
}
#[macro_export]
macro_rules! args {($($value:expr),* $(,)?)=>{&mut[$($crate::cuda::Arg::from($value)),*]};}
pub struct Module {
    raw: usize,
    context: Arc<Context>,
    functions: HashMap<String, usize>,
}
impl Module {
    pub fn new(context: Arc<Context>, source: &str) -> Result<Self> {
        context.activate()?;
        let source = CString::new(source)?;
        let name = CString::new("ddw.cu")?;
        let mut program = ptr::null_mut();
        unsafe {
            ensure!(
                nvrtcCreateProgram(
                    &mut program,
                    source.as_ptr(),
                    name.as_ptr(),
                    0,
                    ptr::null(),
                    ptr::null()
                ) == 0,
                "NVRTC program creation failed"
            );
            let options = [
                CString::new("--std=c++17")?,
                CString::new(format!("--gpu-architecture={}", context.arch))?,
            ];
            let pointers: Vec<_> = options.iter().map(|s| s.as_ptr()).collect();
            let status = nvrtcCompileProgram(program, 2, pointers.as_ptr());
            if status != 0 {
                let mut size = 0;
                nvrtcGetProgramLogSize(program, &mut size);
                let mut log = vec![0u8; size];
                nvrtcGetProgramLog(program, log.as_mut_ptr().cast());
                nvrtcDestroyProgram(&mut program);
                bail!("NVRTC {status}: {}", String::from_utf8_lossy(&log));
            }
            let mut size = 0;
            let status = nvrtcGetPTXSize(program, &mut size);
            if status != 0 {
                nvrtcDestroyProgram(&mut program);
                bail!("NVRTC PTX size error {status}");
            }
            let mut ptx = vec![0u8; size];
            let status = nvrtcGetPTX(program, ptx.as_mut_ptr().cast());
            nvrtcDestroyProgram(&mut program);
            ensure!(status == 0, "NVRTC PTX error {status}");
            let mut raw = ptr::null_mut();
            check(cuModuleLoadData(&mut raw, ptx.as_ptr().cast()))?;
            Ok(Self {
                raw: raw as usize,
                context,
                functions: HashMap::new(),
            })
        }
    }
    pub fn launch(
        &mut self,
        name: &str,
        blocks: usize,
        threads: u32,
        args: &mut [Arg],
    ) -> Result<()> {
        ensure!(
            blocks > 0 && blocks <= u32::MAX as usize,
            "invalid launch size"
        );
        self.context.activate()?;
        let function = if let Some(value) = self.functions.get(name) {
            *value
        } else {
            let c_name = CString::new(name)?;
            let mut value = ptr::null_mut();
            unsafe {
                check(cuModuleGetFunction(
                    &mut value,
                    self.raw as Handle,
                    c_name.as_ptr(),
                ))?;
            }
            self.functions.insert(name.into(), value as usize);
            value as usize
        };
        let mut pointers: Vec<*mut c_void> = args
            .iter_mut()
            .map(|a| match a {
                Arg::U(v) => (v as *mut u64).cast(),
                Arg::I(v) => (v as *mut i32).cast(),
                Arg::F(v) => (v as *mut f64).cast(),
            })
            .collect();
        unsafe {
            check(cuLaunchKernel(
                function as Handle,
                blocks as u32,
                1,
                1,
                threads,
                1,
                1,
                0,
                ptr::null_mut(),
                pointers.as_mut_ptr(),
                ptr::null_mut(),
            ))
        }
    }
}
impl Drop for Module {
    fn drop(&mut self) {
        let _ = self.context.activate();
        unsafe {
            cuModuleUnload(self.raw as Handle);
        }
    }
}
pub(crate) fn read_u64(context: &Context, pointer: u64) -> Result<u64> {
    context.activate()?;
    let mut value = 0u64;
    unsafe {
        check(cuMemcpyDtoH_v2((&mut value as *mut u64).cast(), pointer, 8))?;
    }
    Ok(value)
}
