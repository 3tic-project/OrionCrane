use super::*;

#[derive(Default)]
pub struct ReusableU32TensorBuffer {
    tensor: Option<Tensor>,
    capacity: usize,
}

#[derive(Default)]
pub struct ReusableTensorBuffer {
    tensor: Option<Tensor>,
    capacity: usize,
    dtype: Option<DType>,
}

impl ReusableU32TensorBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        self.tensor = None;
        self.capacity = 0;
    }

    pub fn upload_1d(&mut self, src: &[u32], device: &Device) -> Result<Tensor> {
        if src.is_empty() {
            candle_core::bail!("ReusableU32TensorBuffer cannot upload an empty slice");
        }
        let needs_alloc = self.tensor.as_ref().map_or(true, |tensor| {
            self.capacity < src.len() || !tensor.device().same_device(device)
        });
        if needs_alloc {
            self.capacity = src.len().next_power_of_two();
            self.tensor = Some(Tensor::zeros((self.capacity,), DType::U32, device)?);
        }

        let src_tensor = Tensor::new(src, device)?;
        let tensor = self
            .tensor
            .as_ref()
            .expect("ReusableU32TensorBuffer must be allocated before upload");
        tensor.slice_set(&src_tensor, 0, 0)?;
        tensor.narrow(0, 0, src.len())
    }
}

impl ReusableTensorBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        self.tensor = None;
        self.capacity = 0;
        self.dtype = None;
    }

    pub fn copy_from(&mut self, src: &Tensor) -> Result<Tensor> {
        let len = src.elem_count();
        if len == 0 {
            candle_core::bail!("ReusableTensorBuffer cannot copy an empty tensor");
        }
        let dtype = src.dtype();
        let device = src.device();
        let needs_alloc = self.tensor.as_ref().map_or(true, |tensor| {
            self.capacity < len || !tensor.device().same_device(device) || self.dtype != Some(dtype)
        });
        if needs_alloc {
            self.capacity = len.next_power_of_two();
            self.dtype = Some(dtype);
            self.tensor = Some(Tensor::zeros((self.capacity,), dtype, device)?);
        }

        let src_flat = src.contiguous()?.flatten_all()?;
        let tensor = self
            .tensor
            .as_ref()
            .expect("ReusableTensorBuffer must be allocated before copy");
        tensor.slice_set(&src_flat, 0, 0)?;
        tensor.narrow(0, 0, len)?.reshape(src.dims().to_vec())
    }
}

#[cfg(feature = "cuda")]
pub fn copy_from_slice_u32(src: &[u32], device: &Device) -> Result<Tensor> {
    Tensor::new(src, device)
}

/// Clone a contiguous f32 tensor — returns a new contiguous copy on the same device.
///
/// For CUDA tensors this is a DtoD copy (no host round-trip).
#[cfg(feature = "cuda")]
pub fn copy_from_tensor_f32(src_tensor: &Tensor) -> Result<Tensor> {
    if src_tensor.dtype() != DType::F32 {
        candle_core::bail!(
            "copy_from_tensor_f32: expected f32 tensor, got {:?}",
            src_tensor.dtype()
        );
    }
    src_tensor.contiguous()
}
