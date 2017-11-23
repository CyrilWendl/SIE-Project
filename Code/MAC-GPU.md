# TensorFlow Mac OS X GPU support
Updated on 22nd of November 2017
GPU installation not successful yet

How to make Tensorflow work on Mac OS X using a GPU?

Respect all the versions (don't use newer ones!)

## 1. Prerequisites

```
brew update
brew install coreutils swig 
brew install bazel
```

## 2. Install CUDA 8.0
[Download Link](https://developer.nvidia.com/cuda-80-ga2-download-archive)


```
export PATH=/Developer/NVIDIA/CUDA-8.0/bin${PATH:+:${PATH}}
export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-8.0/lib\
 ${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}
 ```
_According to [this tutorial](https://srikanthpagadala.github.io/notes/2016/11/07/enable-gpu-support-for-tensorflow-on-macos), puts everything in `/usr/local/cuda`_

`# sudo cp /Developer/NVIDIA/CUDA-8.0/ /usr/local/cuda/`

[Downoad cuDNN](https://developer.nvidia.com/cudnn) for CUDA 8.0, MacOS X
Extract it, then move everything to `/usr/local/cuda/`:

```
sudo tar xzvf ~/Downloads/cudnn-8.0-osx-x64-v6.0.tgz
sudo mv -v cuda/lib/libcudnn* /usr/local/cuda/lib
sudo mv -v cuda/include/cudnn.h /usr/local/cuda/include
```

Add the following lines to `~/.bash_profile`:
```
# CUDO, GPU support for TF
export PATH=/usr/local/cuda/
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH
export CUDA_ROOT=$PATH
export LIBRARY_PATH=$CUDA_ROOT/lib:$CUDA_ROOT/lib64:$LIBRARY_PATH
```

[Download XCode 7.2](https://developer.apple.com/download/more/) and make it the default version:
```
mkdir /Applications/XCode7.2.1/
```
Put Xcode.app in this directory, then run:
```
sudo xcode-select -s /Applications/XCode7.2.1/Xcode.app/
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

## 3. Clone the TensorFlow repository

Clone the repo:

`$ git clone https://github.com/tensorflow/tensorflow `

Checkout the right branch
```bash
$ cd tensorflow
$ git checkout 1.1 # >1.2 doesn't support GPU
```

## 4. Configure and build TensorFlow

## 5. Verify GPU is showing up
```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
``` 
