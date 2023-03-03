
<div>
    <a href="https://colab.research.google.com/drive/1ojtbb9nnMJ1oyj_rz0A79o0nKC3CccdI"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>
<br>



# Nuts And Bolts Tracking

Multi-Object Tracking on Nuts&Bolts Dataset with [YOLOv5](https://github.com/ultralytics/yolov5) + [SORT](https://github.com/insight-platform/Similari)

## Description



Basic MOT with seperate detection and tracking parts.

Project relies on official YOLOv5 algorithm and Rust's Similari library for SORT algorithm.

All detection related information can be found under runs/exp4

In the following setup, tracking execution for only first frame is above 1/30 seconds.




## Getting Started

### Dependencies



* All dependencies can be installed except Similari with following:

```
conda env create -f environment.yaml
```


* Project built and run on WSL2, details are as follows:

```
WSL sürümü: 1.1.3.0
Çekirdek sürümü: 5.15.90.1
WSLg sürümü: 1.0.49
MSRDC sürümü: 1.2.3770
Direct3D sürümü: 1.608.2-61064218
DXCore sürümü: 10.0.25131.1002-220531-1700.rs-onecore-base2-hyp
Windows sürümü: 10.0.22621.1265
```


* Output of collect_env.py is as follows:

```
Collecting environment information...
PyTorch version: 1.13.1
Is debug build: False
CUDA used to build PyTorch: 11.7
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.1 LTS (x86_64)
GCC version: (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.9.16 | packaged by conda-forge | (main, Feb  1 2023, 21:39:03)  [GCC 11.3.0] (64-bit runtime)
Python platform: Linux-5.15.90.1-microsoft-standard-WSL2-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 11.7.99
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce MX450
Nvidia driver version: 528.49
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

Versions of relevant libraries:
[pip3] numpy==1.23.5
[pip3] pytorch-lightning==1.9.3
[pip3] torch==1.13.1
[pip3] torch-tb-profiler==0.4.1
[pip3] torchaudio==0.13.1
[pip3] torchensemble==0.1.9
[pip3] torchmetrics==0.11.1
[pip3] torchreid==0.2.5
[pip3] torchsummary==1.5.1
[pip3] torchvision==0.14.1
[conda] blas                      1.0                         mkl  
[conda] libblas                   3.9.0            12_linux64_mkl    conda-forge
[conda] libcblas                  3.9.0            12_linux64_mkl    conda-forge
[conda] liblapack                 3.9.0            12_linux64_mkl    conda-forge
[conda] liblapacke                3.9.0            12_linux64_mkl    conda-forge
[conda] mkl                       2021.4.0           h06a4308_640  
[conda] mkl-service               2.4.0            py39h7e14d7c_0    conda-forge
[conda] mkl_fft                   1.3.1            py39h0c7bc48_1    conda-forge
[conda] mkl_random                1.2.2            py39hde0f152_0    conda-forge
[conda] numpy                     1.23.5           py39h14f4228_0  
[conda] numpy-base                1.23.5           py39h31eccc5_0  
[conda] pytorch                   1.13.1          py3.9_cuda11.7_cudnn8.5.0_0    pytorch
[conda] pytorch-cuda              11.7                 h67b0de4_1    pytorch
[conda] pytorch-lightning         1.9.3              pyhd8ed1ab_0    conda-forge
[conda] pytorch-mutex             1.0                        cuda    pytorch
[conda] torch-tb-profiler         0.4.1                    pypi_0    pypi
[conda] torchaudio                0.13.1               py39_cu117    pytorch
[conda] torchensemble             0.1.9                    pypi_0    pypi
[conda] torchmetrics              0.11.1             pyhd8ed1ab_0    conda-forge
[conda] torchreid                 0.2.5                    pypi_0    pypi
[conda] torchsummary              1.5.1                    pypi_0    pypi
[conda] torchvision               0.14.1               py39_cu117    pytorch
```



### Instructions


* For downloading challenge and making data ready for YOLO format:

```
bash download_challenge.sh
```


* For training:

```
bash download_weights.sh
./train.sh --weights ${INITIAL_WEIGHT_PATH} --img 640 --epochs 25 --batch 3 --data ${DATA_YAML_PATH} --project . --hyp ${HYPERPARAMETERS_PATH}
```


* For Inference

```
python yolov5/export.py --weights ${TORCH_FINAL_WEIGHTS} --include engine --device 0
python main.py --weights ${EXPORTED_WEIGHTS} --input ${INPUT_VIDEO}
```



### Executing program


All process can be simulated with following [Colab Notebook](https://colab.research.google.com/drive/1ojtbb9nnMJ1oyj_rz0A79o0nKC3CccdI)

Private keys on Colab is for this repo only.


## Authors


* Serkan Şatak - serkansatak1@gmail.com


## Acknowledgments


Inspiration, code snippets, etc.

* [High-Performance Python SORT Tracker](https://medium.com/inside-in-sight/high-performance-python-sort-tracker-225c2b507562)
* [Similari](https://github.com/insight-platform/Similari)
* [YOLOv5](https://github.com/ultralytics/yolov5)
* [COCO2YOLO](https://github.com/alexmihalyk23/COCO2YOLO/blob/master/COCO2YOLO.py)