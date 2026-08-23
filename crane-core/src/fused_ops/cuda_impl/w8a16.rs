use super::*;
use candle_core::{CpuStorage, CustomOp2, CustomOp3};

/// Strict weight-only INT8 linear operation.
///
/// `x` is BF16 with shape `[..., K]`, `weight` is U8 `[N, K]` containing
/// signed q8 values biased by 128, and `scales` is F32 `[N]`.
struct W8A16LinearOp {
    m: usize,
    k: usize,
    n: usize,
    output_shape: Shape,
    split_k: usize,
}

impl W8A16LinearOp {
    fn validate_layouts(&self, x: &Layout, weight: &Layout, scales: &Layout) -> Result<()> {
        if x.shape().elem_count() != self.m * self.k {
            candle_core::bail!("w8a16_linear: invalid activation shape")
        }
        if weight.shape().dims() != [self.n, self.k] {
            candle_core::bail!(
                "w8a16_linear: weight must have shape [{}, {}], got {:?}",
                self.n,
                self.k,
                weight.shape().dims()
            )
        }
        if scales.shape().dims() != [self.n] {
            candle_core::bail!(
                "w8a16_linear: scales must have shape [{}], got {:?}",
                self.n,
                scales.shape().dims()
            )
        }
        if x.contiguous_offsets().is_none()
            || weight.contiguous_offsets().is_none()
            || scales.contiguous_offsets().is_none()
        {
            candle_core::bail!("w8a16_linear: all inputs must be contiguous")
        }
        Ok(())
    }
}

impl CustomOp3 for W8A16LinearOp {
    fn name(&self) -> &'static str {
        "w8a16-linear"
    }

    fn cpu_fwd(
        &self,
        x_storage: &CpuStorage,
        x_layout: &Layout,
        weight_storage: &CpuStorage,
        weight_layout: &Layout,
        scales_storage: &CpuStorage,
        scales_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        self.validate_layouts(x_layout, weight_layout, scales_layout)?;
        let (x_start, x_end) = x_layout.contiguous_offsets().unwrap();
        let (w_start, w_end) = weight_layout.contiguous_offsets().unwrap();
        let (s_start, s_end) = scales_layout.contiguous_offsets().unwrap();
        let (x, weight, scales) = match (x_storage, weight_storage, scales_storage) {
            (CpuStorage::BF16(x), CpuStorage::U8(weight), CpuStorage::F32(scales)) => (
                &x[x_start..x_end],
                &weight[w_start..w_end],
                &scales[s_start..s_end],
            ),
            _ => candle_core::bail!(
                "w8a16_linear expects BF16 activations, U8 weights, and F32 scales"
            ),
        };

        let mut output = vec![half::bf16::ZERO; self.m * self.n];
        for row in 0..self.m {
            for out_col in 0..self.n {
                let mut sum = 0.0f32;
                for inner in 0..self.k {
                    let activation = x[row * self.k + inner].to_f32();
                    let q = i32::from(weight[out_col * self.k + inner]) - 128;
                    let dequantized = half::bf16::from_f32(q as f32 * scales[out_col]).to_f32();
                    sum += activation * dequantized;
                }
                output[row * self.n + out_col] = half::bf16::from_f32(sum);
            }
        }
        Ok((CpuStorage::BF16(output), self.output_shape.clone()))
    }

    fn cuda_fwd(
        &self,
        x_storage: &CudaStorage,
        x_layout: &Layout,
        weight_storage: &CudaStorage,
        weight_layout: &Layout,
        scales_storage: &CudaStorage,
        scales_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        self.validate_layouts(x_layout, weight_layout, scales_layout)?;
        if self.k == 0 || self.n == 0 || self.m == 0 {
            candle_core::bail!("w8a16_linear: dimensions must be non-zero")
        }
        if self.k > i32::MAX as usize || self.n > i32::MAX as usize || self.m > i32::MAX as usize {
            candle_core::bail!("w8a16_linear: dimensions exceed CUDA kernel limits")
        }

        let dev = x_storage.device();
        let (x_start, x_end) = x_layout.contiguous_offsets().unwrap();
        let (w_start, w_end) = weight_layout.contiguous_offsets().unwrap();
        let (s_start, s_end) = scales_layout.contiguous_offsets().unwrap();
        let m = self.m as i32;
        let k = self.k as i32;
        let n = self.n as i32;

        let slice = match (
            &x_storage.slice,
            &weight_storage.slice,
            &scales_storage.slice,
        ) {
            (
                CudaStorageSlice::BF16(x),
                CudaStorageSlice::U8(weight),
                CudaStorageSlice::F32(scales),
            ) => {
                let x = x.slice(x_start..x_end);
                let weight = weight.slice(w_start..w_end);
                let scales = scales.slice(s_start..s_end);
                let output = unsafe { dev.alloc::<half::bf16>(self.m * self.n)? };
                if self.split_k == 1 {
                    let func = load_func!(dev, "w8a16_linear_bf16")?;
                    let cfg = LaunchConfig {
                        grid_dim: (self.n.div_ceil(16) as u32, self.m.div_ceil(16) as u32, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut builder = func.builder();
                    builder.arg(&x);
                    builder.arg(&weight);
                    builder.arg(&scales);
                    builder.arg(&output);
                    builder.arg(&m);
                    builder.arg(&k);
                    builder.arg(&n);
                    unsafe { builder.launch(cfg) }.w()?;
                } else {
                    let split_k = self.split_k as i32;
                    let partial = unsafe { dev.alloc::<f32>(self.split_k * self.m * self.n)? };
                    let gemm = load_func!(dev, "w8a16_linear_bf16_splitk")?;
                    let gemm_cfg = LaunchConfig {
                        grid_dim: (
                            self.n.div_ceil(16) as u32,
                            self.m.div_ceil(16) as u32,
                            self.split_k as u32,
                        ),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut builder = gemm.builder();
                    builder.arg(&x);
                    builder.arg(&weight);
                    builder.arg(&scales);
                    builder.arg(&partial);
                    builder.arg(&m);
                    builder.arg(&k);
                    builder.arg(&n);
                    builder.arg(&split_k);
                    unsafe { builder.launch(gemm_cfg) }.w()?;

                    let elements = (self.m * self.n) as i32;
                    let reduce = load_func!(dev, "w8a16_splitk_reduce_bf16")?;
                    let reduce_cfg = LaunchConfig {
                        grid_dim: ((self.m * self.n).div_ceil(256) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut builder = reduce.builder();
                    builder.arg(&partial);
                    builder.arg(&output);
                    builder.arg(&elements);
                    builder.arg(&split_k);
                    unsafe { builder.launch(reduce_cfg) }.w()?;
                }
                CudaStorageSlice::BF16(output)
            }
            _ => candle_core::bail!(
                "w8a16_linear expects BF16 activations, U8 weights, and F32 scales"
            ),
        };

        Ok((
            CudaStorage {
                slice,
                device: dev.clone(),
            },
            self.output_shape.clone(),
        ))
    }
}

/// Applies a strict W8A16 linear operation without bias.
pub fn w8a16_linear(x: &Tensor, weight: &Tensor, scales: &Tensor) -> Result<Tensor> {
    let k = *x
        .dims()
        .last()
        .ok_or_else(|| candle_core::Error::Msg("w8a16_linear expects rank >= 1".into()))?;
    let m = x.elem_count() / k;
    let n = weight.dim(0)?;
    let split_k = if m <= 1 {
        if k >= 4096 {
            16
        } else if n >= 8192 {
            4
        } else if n >= 4096 {
            8
        } else {
            8
        }
    } else if m <= 16 {
        if n >= 8192 {
            2
        } else if k >= 4096 {
            16
        } else if n >= 4096 {
            4
        } else {
            8
        }
    } else if m <= 32 {
        if n >= 8192 {
            4
        } else if k >= 4096 {
            4
        } else if n >= 4096 {
            2
        } else {
            4
        }
    } else {
        1
    };
    w8a16_linear_with_split(x, weight, scales, split_k)
}

struct W8A16DequantizeOp {
    n: usize,
    k: usize,
}

impl CustomOp2 for W8A16DequantizeOp {
    fn name(&self) -> &'static str {
        "w8a16-dequantize"
    }

    fn cpu_fwd(
        &self,
        weight_storage: &CpuStorage,
        weight_layout: &Layout,
        scales_storage: &CpuStorage,
        scales_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (w_start, w_end) = weight_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("weight must be contiguous".into()))?;
        let (s_start, s_end) = scales_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("scales must be contiguous".into()))?;
        let (weight, scales) = match (weight_storage, scales_storage) {
            (CpuStorage::U8(weight), CpuStorage::F32(scales)) => {
                (&weight[w_start..w_end], &scales[s_start..s_end])
            }
            _ => candle_core::bail!("w8a16_dequantize expects U8 weights and F32 scales"),
        };
        let output = weight
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let q = i32::from(*value) - 128;
                half::bf16::from_f32(q as f32 * scales[index / self.k])
            })
            .collect::<Vec<_>>();
        Ok((CpuStorage::BF16(output), Shape::from((self.n, self.k))))
    }

    fn cuda_fwd(
        &self,
        weight_storage: &CudaStorage,
        weight_layout: &Layout,
        scales_storage: &CudaStorage,
        scales_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (w_start, w_end) = weight_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("weight must be contiguous".into()))?;
        let (s_start, s_end) = scales_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("scales must be contiguous".into()))?;
        let dev = weight_storage.device();
        let output = unsafe { dev.alloc::<half::bf16>(self.n * self.k)? };
        match (&weight_storage.slice, &scales_storage.slice) {
            (CudaStorageSlice::U8(weight), CudaStorageSlice::F32(scales)) => {
                let weight = weight.slice(w_start..w_end);
                let scales = scales.slice(s_start..s_end);
                let func = load_func!(dev, "w8a16_dequantize_bf16")?;
                let cfg = LaunchConfig {
                    grid_dim: ((self.n * self.k).div_ceil(256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let elements = (self.n * self.k) as i32;
                let k = self.k as i32;
                let mut builder = func.builder();
                builder.arg(&weight);
                builder.arg(&scales);
                builder.arg(&output);
                builder.arg(&elements);
                builder.arg(&k);
                unsafe { builder.launch(cfg) }.w()?;
            }
            _ => candle_core::bail!("w8a16_dequantize expects U8 weights and F32 scales"),
        }
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(output),
                device: dev.clone(),
            },
            Shape::from((self.n, self.k)),
        ))
    }
}

/// Restores a per-output-channel W8 weight to BF16 for the large-M cuBLAS path.
pub fn w8a16_dequantize(weight: &Tensor, scales: &Tensor) -> Result<Tensor> {
    let (n, k) = weight.dims2()?;
    if weight.dtype() != DType::U8 || scales.dtype() != DType::F32 || scales.dims() != [n] {
        candle_core::bail!("w8a16_dequantize expects U8 [N,K] and F32 [N]")
    }
    if !weight.device().same_device(scales.device()) {
        candle_core::bail!("w8a16_dequantize: inputs must be on the same device")
    }
    weight
        .contiguous()?
        .apply_op2_no_bwd(&scales.contiguous()?, &W8A16DequantizeOp { n, k })
}

fn w8a16_linear_with_split(
    x: &Tensor,
    weight: &Tensor,
    scales: &Tensor,
    requested_split_k: usize,
) -> Result<Tensor> {
    let k = *x
        .dims()
        .last()
        .ok_or_else(|| candle_core::Error::Msg("w8a16_linear expects rank >= 1".into()))?;
    let (n, weight_k) = weight.dims2()?;
    if k != weight_k {
        candle_core::bail!("w8a16_linear: activation K {k} does not match weight K {weight_k}")
    }
    if scales.dims() != [n] {
        candle_core::bail!("w8a16_linear: scales must have shape [{n}]")
    }
    if k % 16 != 0 {
        candle_core::bail!("w8a16_linear: K must be a multiple of 16, got {k}")
    }
    if x.dtype() != DType::BF16 || weight.dtype() != DType::U8 || scales.dtype() != DType::F32 {
        candle_core::bail!("w8a16_linear expects BF16 activations, U8 weights, and F32 scales")
    }
    if !x.device().same_device(weight.device()) || !x.device().same_device(scales.device()) {
        candle_core::bail!("w8a16_linear: all inputs must be on the same device")
    }

    let m = x.elem_count() / k;
    let k_tiles = k / 16;
    let requested_split_k = requested_split_k.max(1).min(k_tiles);
    let tiles_per_split = k_tiles.div_ceil(requested_split_k);
    // Avoid launching an empty final split for non-power-of-two K tile counts.
    let split_k = k_tiles.div_ceil(tiles_per_split);
    let mut output_dims = x.dims().to_vec();
    *output_dims.last_mut().unwrap() = n;
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    let scales = scales.contiguous()?;
    x.apply_op3_no_bwd(
        &weight,
        &scales,
        &W8A16LinearOp {
            m,
            k,
            n,
            output_shape: Shape::from_dims(&output_dims),
            split_k,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Module};
    use std::time::Instant;

    fn test_case(m: usize, k: usize, n: usize) -> Result<()> {
        let device = Device::new_cuda(0)?;
        let x_values = (0..m * k)
            .map(|i| half::bf16::from_f32(((i * 17 % 101) as f32 - 50.0) / 37.0))
            .collect::<Vec<_>>();
        let q_values = (0..n * k)
            .map(|i| (((i * 29 + 7) % 255) as u8).max(1))
            .collect::<Vec<_>>();
        let scales_values = (0..n)
            .map(|i| 0.001 + (i % 13) as f32 * 0.0007)
            .collect::<Vec<_>>();

        let x_cpu = Tensor::from_vec(x_values, (m, k), &Device::Cpu)?;
        let q_cpu = Tensor::from_vec(q_values, (n, k), &Device::Cpu)?;
        let scales_cpu = Tensor::from_vec(scales_values, n, &Device::Cpu)?;
        let expected = w8a16_linear(&x_cpu, &q_cpu, &scales_cpu)?;
        let actual = w8a16_linear(
            &x_cpu.to_device(&device)?,
            &q_cpu.to_device(&device)?,
            &scales_cpu.to_device(&device)?,
        )?
        .to_device(&Device::Cpu)?;

        let expected = expected.flatten_all()?.to_vec1::<half::bf16>()?;
        let actual = actual.flatten_all()?.to_vec1::<half::bf16>()?;
        let max_abs_error = expected
            .iter()
            .zip(actual.iter())
            .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_error <= 0.0625,
            "m={m} k={k} n={n}: max_abs_error={max_abs_error}"
        );
        Ok(())
    }

    #[test]
    fn cuda_matches_cpu_for_decode_and_batch_shapes() -> Result<()> {
        for m in [1, 7, 16, 31, 32] {
            test_case(m, 32, 48)?;
        }
        test_case(7, 80, 48)?;
        Ok(())
    }

    #[test]
    fn cuda_dequantize_matches_cpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let (n, k) = (48usize, 32usize);
        let q_values = (0..n * k)
            .map(|i| (((i * 19 + 11) % 255) as u8).max(1))
            .collect::<Vec<_>>();
        let scale_values = (0..n)
            .map(|i| 0.001 + (i % 17) as f32 * 0.0003)
            .collect::<Vec<_>>();
        let q_cpu = Tensor::from_vec(q_values, (n, k), &Device::Cpu)?;
        let scales_cpu = Tensor::from_vec(scale_values, n, &Device::Cpu)?;
        let expected = w8a16_dequantize(&q_cpu, &scales_cpu)?;
        let actual = w8a16_dequantize(&q_cpu.to_device(&device)?, &scales_cpu.to_device(&device)?)?
            .to_device(&Device::Cpu)?;
        assert_eq!(
            expected.flatten_all()?.to_vec1::<half::bf16>()?,
            actual.flatten_all()?.to_vec1::<half::bf16>()?
        );
        Ok(())
    }

    #[test]
    #[ignore = "manual GPU microbenchmark"]
    fn benchmark_qwen3_1_7b_projection_shapes() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let shapes = [
            ("qkv", 2048usize, 4096usize),
            ("o_proj", 2048, 2048),
            ("gate_up", 2048, 12288),
            ("down_proj", 6144, 2048),
        ];

        for m in [1usize, 16, 32] {
            for (name, k, n) in shapes {
                let x = Tensor::ones((m, k), DType::BF16, &device)?;
                let q_values = (0..n * k)
                    .map(|i| (((i * 29 + 7) % 255) as u8).max(1))
                    .collect::<Vec<_>>();
                let scale = 0.002f32;
                let bf16_values = q_values
                    .iter()
                    .map(|q| half::bf16::from_f32((i32::from(*q) - 128) as f32 * scale))
                    .collect::<Vec<_>>();
                let qweight = Tensor::from_vec(q_values, (n, k), &device)?;
                let scales = Tensor::from_vec(vec![scale; n], n, &device)?;
                let bf16_weight = Tensor::from_vec(bf16_values, (n, k), &device)?;
                let bf16_linear = candle_nn::Linear::new(bf16_weight, None);

                for _ in 0..3 {
                    let _ = bf16_linear.forward(&x)?;
                }
                device.synchronize()?;

                let iterations = 50;
                let start = Instant::now();
                for _ in 0..iterations {
                    let _ = bf16_linear.forward(&x)?;
                }
                device.synchronize()?;
                let bf16_us = start.elapsed().as_secs_f64() * 1e6 / iterations as f64;
                for split_k in [1usize, 2, 4, 8, 16, 32] {
                    for _ in 0..3 {
                        let _ = w8a16_linear_with_split(&x, &qweight, &scales, split_k)?;
                    }
                    device.synchronize()?;
                    let start = Instant::now();
                    for _ in 0..iterations {
                        let _ = w8a16_linear_with_split(&x, &qweight, &scales, split_k)?;
                    }
                    device.synchronize()?;
                    let w8_us = start.elapsed().as_secs_f64() * 1e6 / iterations as f64;
                    eprintln!(
                        "W8A16_BENCH m={m:>2} op={name:<9} k={k:<4} n={n:<5} split={split_k:<2} w8_us={w8_us:>8.2} bf16_us={bf16_us:>8.2} speedup={:.3}",
                        bf16_us / w8_us
                    );
                }
            }
        }
        Ok(())
    }
}
