# EmVision-YoLoV8-Benchmark

Ultralytics의 YoloV8를 수행합니다.

## 시험 환경 구성

### 1. Python 가상 환경 구성

파이썬 가상환경 구성을 통해 리소스관리와 환경 관리를 용이하게 할 수 있습니다.

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

python3 -m venv .pyenv --system-site-packages
source .pyenv/bin/activate
python -m pip install --upgrade pip
```

### 2. PyTorch 및 TorchVision 설치

```bash
# PyTorch 설치
source .pyenv/bin/activate
python3 -m pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

# TorchVision 설치
sudo apt-get install -y git libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev python3-setuptools
python3 -m pip install --upgrade pip packaging
export BUILD_VERSION=0.16.1
git clone --branch v0.16.1 https://github.com/pytorch/vision torchvision --depth 1 && cd torchvision
pip install -e . && cd ..
rm -rf torchvision
```

### 3. ultralytics 설치

```bash
source .pyenv/bin/activate
pip install -U ultralytics
```

### 4. onnxruntime-gpu 설치

```bash
source .pyenv/bin/activate
pip install onnxruntime/onnxruntime_gpu-1.16.3-cp38-cp38-linux_aarch64.whl
```

## 벤치마크 수행

```bash
source .pyenv/bin/activate
python3 benchmark.py
```

벤치마크 결과는 `benchmark_{모듈명}.md`에 저장됩니다.

| 모듈명 | 제품명 |
| --- | --- |
| Orin Nano 4GB | **Emvision ON^4^** |
| Orin Nano 8GB | **Emvision ON^8^** |
| Orin NX 8GB | **Emvision OX^8^** |
| Orin NX 16GB | **Emvision OX^16^** |
