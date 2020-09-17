# YOLOv3_TensorRT
A TensorRT implementation of YOLOv3
### Environment

- Ubuntu 18.04
- TensorRT-6.0..1.5
- python==2.7 (only for "yolov3_to_onnx.py")
- onnx==1.2.1 (important)
- python==3.6.5
- torch==1.6.0+cu101
- pycuda
- opencv-python

### Getting Started

#### Install TensorRT

1. Install TensorRT from  https://developer.nvidia.com/tensorrt.

2. Add TensorRT to environment variables.

3. Install TensorRT Python API.

   ```shell
   $ cd TensorRT-6.x.x.x/python
   $ sudo pip3 install tensorrt-6.x.x.x-cp3x-none-linux_x86_64.whl
   ```

4. Install UFF.

   ```shell
   $ cd TensorRT-6.x.x.x/uff
   $ sudo pip3 install uff-0.6.5-py2.py3-none-any.whl
   ```

5. Install graphsurgeon.

   ```shell
   $ cd TensorRT-6.x.x.x/graphsurgeon
   $ sudo pip3 install graphsurgeon-0.4.1-py2.py3-none-any.whl
   ```

#### Convert weights to onnx

1. Download weights file from https://pjreddie.com/darknet/yolo/ and place in "./models".
2. Run "yolov3_to_onnx.py" with **Python 2.x** and **onnx==1.2.1** (only work for this version).

#### Convert onnx to trt and run

1. Run "video_demo.py", It may take a long time to run for the first time.

### Performance

| Model      | GPU        | Inference Time |
| ---------- | ---------- | -------------- |
| YOLOv3-416 | RTX 2080TI | 10.21ms        |
| YOLOv3-608 | RTX 2080TI | 16.01ms        |

