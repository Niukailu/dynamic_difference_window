fn main() {
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    // Link against the toolkit stub; the driver provides libcuda.so.1 at runtime.
    println!("cargo:rustc-link-search=native={cuda}/lib64/stubs");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=nvrtc");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{cuda}/lib64");
}
