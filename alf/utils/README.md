# Pytorch eager mode -> ONNX/TensorRT cheatsheet

## ONNX
### `torch.onnx.export(opset_version)`
Suggest to set `opset_version=12`.

### Pytorch resize functions/modules
*Blacklist*:

- `torchvision.transforms.Resize`
- `torch.nn.UpsamplingBilinear2d`
- `torch.nn.UpsamplingNearest2d`
- `torch.nn.functional.upsample_bilinear`
- `torch.nn.functional.upsample_nearest`

*Reason*: Bilinear resizing is not supported by ONNX opset_version>=12. Nearest resizing is supported, but usually producing segfaults due to the `Slice` operator.

*Solution*: use our own custom NN/bilinear resizer, built from pytorch functions.

### Tensor repeating
*Blacklist*:

- `torch.repeat`

*Reason*: ONNX will use `Slice` again for this torch function, causing segfault.

*Solution*: use `torch.expand` instead. However, this only applies to repeating singular dims.

### Tensor partial assignment
Do not use index slicing assignment such as `x[idx] = y`, as it will trigger the `Slice` segfault.

*Solution*: use `torch.scatter` or `Tensor.scatter_`.

## TensorRT
### Data type support
*Blacklist*:

- uint8
- int16
- float64

*Solution*:

- avoid int16.
- Convert uint8 to int32, float64 to float32 before running the TensorRT engine.
- Avoid creating intermediate tensor nodes of dtype uint8, int16, or float64 for the deployment code branch

### Dynamic data shape
TensorRT does support dynamic intermediate data shape. For example, in one forward inference some intermediate node has a
shape of ``[2,...]`` while in another forward the same node has a shape of ``[1,...]`` (maybe as a result of index selection).
However, whenever it encounters a new shape for the first time, a **considerable** amount (tens of seconds) of computational
time will be spent. So in order to avoid any unexpected surprising performance drop during inference, the code should be
written in a way that each node has a constant shape regardless of the input.

For example,

```python
# Below is unfriendly to TensorRT
is_first = timestep.is_first() # [B]
obs = timestep.observation[is_first] # [K]; dynamic shape
new_obs[is_first] = torch.zeros_like(obs)

# A better implementation
new_obs = torch.where(timestep.is_first(),
                      torch.zeros_like(timestpe.observation),
                      timestep.observation)
```

If you have to use a dynamic shape, you can choose to use the CUDA backend by setting the environment variable

```bash
ORT_ONNX_BACKEND_EXCLUDE_PROVIDERS=TensorrtExecutionProvider
```

### `rsample()`
TensorRT will report "ERROR: Network must have at least one output" when there is any `rsample()` in the eager mode code. Fortunately, for inference we can usually use the distribution's mode or `sample()` instead to avoid this issue.

## Common issues
There is some known side effect on CUDA/GPU when importing ``tensorrt_utils.py``. It is crucial to make sure
that this module is never imported during training, but only imported for inference when necessary.

If imported during training, the typical issues can be:

1. GPU 0 consumes an extra abnormal amount of memory, leading to CUDA out-of-mem issue.
2. For multi-gpu training, sometimes there will be an error such as "Duplicate GPU detected : rank 1 and rank 0 both on CUDA device 17000" after actual training starts (rollout is fine).
