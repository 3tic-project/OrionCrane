fn main() {
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-changed=kernels/fused_ops/common.cuh");
    println!("cargo:rerun-if-changed=kernels/fused_ops/basic_ops.cuh");
    println!("cargo:rerun-if-changed=kernels/fused_ops/sampling.cuh");
    println!("cargo:rerun-if-changed=kernels/fused_ops/paged_kv.cuh");
    println!("cargo:rerun-if-changed=kernels/fused_ops/topk.cuh");
    println!("cargo:rerun-if-changed=kernels/fused_ops/rope.cuh");
    println!("cargo:rerun-if-changed=kernels/fused_ops/w8a16.cuh");
    println!("cargo:rerun-if-changed=build.rs");

    // Only compile CUDA kernels when the cuda feature is enabled.
    #[cfg(feature = "cuda")]
    {
        use std::env;
        use std::path::PathBuf;

        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

        let builder = bindgen_cuda::Builder::default()
            .kernel_paths_glob("kernels/**/*.cu")
            .arg("--expt-relaxed-constexpr")
            .arg("-std=c++17")
            .arg("-O3");

        let bindings = builder.build_ptx().expect("Failed to compile CUDA kernels");
        bindings
            .write(out_dir.join("crane_kernels_ptx.rs"))
            .expect("Failed to write PTX bindings");
    }
}
