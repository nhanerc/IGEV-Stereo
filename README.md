This repo is forked from https://github.com/gangweiX/IGEV with some modifications in order to run onnx models on Jetson Jetpack 5.

To convert the model from pytorch to onnx, run the following command:

```bash
python utils/onnx_converter.py --restore_ckpt ./pretrained_models/middlebury.pth --save_onnx_path ./pretrained_models/middlebury.onnx --img_size 608 800 --iter 32
```

```
trtexec --workspace=10240 --verbose --onnx=middlebury.onnx --saveEngine=middlebury.trt
trtexec --workspace=10240 --iterations=100 --avgRuns=10 --verbose --loadEngine=middlebury.trt
```

32: Latency: min = 2988.92 ms, max = 3054.5 ms, mean = 3024.96 ms, median = 3025.03 ms, percentile(99%) = 3054.5 ms
16: Latency: min = 1704.72 ms, max = 1743.48 ms, mean = 1723.09 ms, median = 1723.12 ms, percentile(99%) = 1743.48 ms
