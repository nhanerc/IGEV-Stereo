This repo is forked from https://github.com/gangweiX/IGEV with some modifications in order to run onnx models on Jetson Jetpack 5.


# Create a conda environment
```bash
conda create -n stereo python=3.8
conda activate stereo
pip install -r requirements.txt
```

# Download the pretrained model
Browse the original repo to download pretrained models and to convert the model from pytorch to onnx, run the following command:
```bash
python utils/onnx_converter.py --restore_ckpt ./pretrained_models/middlebury.pth --save_onnx_path ./pretrained_models/middlebury.onnx --img_size 608 800 --iter 32
```

On Jetson Jetpack 5, run the following command to generate the trt engine:
```bash
trtexec --workspace=10240 --verbose --onnx=middlebury.onnx --saveEngine=middlebury.trt
# trtexec --workspace=10240 --iterations=100 --avgRuns=10 --verbose --loadEngine=middlebury.trt
```

Or visualize the result by using `viz.ipynb`
