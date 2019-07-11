# PCN in Pytorch


Progressive Calibration Networks (PCN) is an accurate rotation-invariant face detector running at real-time speed on CPU. This is an implementation for PCN.

This is a pytorch implementation version of the [original repo](https://github.com/Jack-CV/FaceKit/tree/master/PCN)

## Getting Started

A separate Python environment is recommended.
+ Python3.5+ (Python3.5, Python3.6 are tested)
+ Pytorch == 1.0
+ opencv4 (opencv3.4.5 is tested also)
+ numpy

install dependences using `pip`
```bash
pip3 install numpy opencv-python
pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision (optional)
```
or install using `conda`
```bash
conda install opencv numpy
conda install pytorch-cpu torchvision-cpu -c pytorch
```

## Usage
```bash
cd pcn
python demo.py path/to/image 
```
or use webcam demo
```bash
python webcam.py
```

## Install
```
cd pcn && pip install .
```

