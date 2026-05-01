use super::*;
use candle_core::cuda_backend::cudarc::driver::{sys, CudaGraph, CudaStream};
use std::sync::Arc;

pub struct CudaGraphExec {
    graph: CudaGraph,
}

// CUDA graph handles are used only by the single engine thread after creation;
// the wrapper deliberately does not implement Sync because graph APIs are not
// safe for concurrent access.
unsafe impl Send for CudaGraphExec {}

impl CudaGraphExec {
    pub fn upload(&self) -> Result<()> {
        self.graph.upload().w()
    }

    pub fn launch(&self) -> Result<()> {
        self.graph.launch().w()
    }
}

pub struct CudaGraphCapture {
    stream: Arc<CudaStream>,
    finished: bool,
}

impl CudaGraphCapture {
    pub fn begin(device: &Device) -> Result<Self> {
        let dev = match device {
            Device::Cuda(dev) => dev,
            _ => candle_core::bail!("CudaGraphCapture requires a CUDA device"),
        };
        let stream = dev.cuda_stream();
        stream
            .begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
            .w()?;
        Ok(Self {
            stream,
            finished: false,
        })
    }

    pub fn end(mut self) -> Result<Option<CudaGraphExec>> {
        self.finished = true;
        let graph = self.stream.end_capture_with_flags(0).w()?;
        Ok(graph.map(|graph| CudaGraphExec { graph }))
    }
}

impl Drop for CudaGraphCapture {
    fn drop(&mut self) {
        debug_assert!(
            self.finished,
            "CudaGraphCapture dropped without end(); the CUDA stream may still be capturing"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    #[test]
    fn captures_and_replays_single_kernel() -> Result<()> {
        let device = Device::new_cuda_with_stream(0)?;
        let dev = match &device {
            Device::Cuda(dev) => dev,
            _ => unreachable!(),
        };
        unsafe { dev.disable_event_tracking() };
        let func = load_func!(dev, "fused_residual_add_bf16")?;
        let stream = dev.cuda_stream();
        let len = 16usize;
        let mut residual = dev.clone_htod(&vec![bf16::from_f32(1.0); len])?;
        let hidden = dev.clone_htod(&vec![bf16::from_f32(2.0); len])?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let len_i32 = len as i32;

        let capture = CudaGraphCapture::begin(&device)?;
        let mut builder = func.builder();
        builder.arg(&mut residual);
        builder.arg(&hidden);
        builder.arg(&len_i32);
        unsafe { builder.launch(cfg) }.w()?;
        let graph = capture
            .end()?
            .ok_or_else(|| candle_core::Error::Msg("empty CUDA graph capture".into()))?;
        graph.upload()?;
        graph.launch()?;
        graph.launch()?;
        stream.synchronize().w()?;

        let out = dev.clone_dtoh(&residual)?;
        assert!(out.iter().all(|value| (value.to_f32() - 5.0).abs() < 0.01));
        Ok(())
    }
}
