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
