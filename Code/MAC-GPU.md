# TensorFlow Mac OS X GPU support
Updated on 22nd of November 2017
GPU installation not successful yet

How to make Tensorflow work on Mac OS X using a GPU?


## 1. Prerequisites

```
brew update
brew install coreutils swig 
brew install bazel
```

## 2. Clone the TensorFlow repository

Clone the repo:

`$ git clone https://github.com/tensorflow/tensorflow `

Checkout the right branch
```bash
$ cd tensorflow
$ git checkout 1.1 # >1.2 doesn't support GPU
```

## 3. Install CUDA 8.0
[Download Link](https://developer.nvidia.com/cuda-80-ga2-download-archive)

According to [this tutorial](https://srikanthpagadala.github.io/notes/2016/11/07/enable-gpu-support-for-tensorflow-on-macos), do everything in `/usr/local/cuda`

`sudo cp /Developer/NVIDIA/CUDA-8.0/ /usr/local/cuda/`

[Downoad cuDNN](https://developer.nvidia.com/cudnn) for CUDA 8.0, MacOS X
Extract it, then move everything to `/usr/local/cuda/`:

```
sudo tar xzvf ~/Downloads/cudnn-8.0-osx-x64-v6.0.tgz
sudo cp -v cuda/lib/libcudnn* /usr/local/cuda/lib
sudo cp -v cuda/include/cudnn.h /usr/local/cuda/include
```

Verify everything is working:

```
# build example
cd /usr/local/cuda/samples
sudo make -C 1_Utilities/deviceQuery

# run it
cd /usr/local/cuda/samples/
./bin/x86_64/darwin/release/deviceQuery
```

## 4. Configure and build TensorFlow

## 5. Verify GPU is showing up
```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
``` 
