# Parallel pytlsd

> currently work in progress, the implementation is not done

Python transparent bindings for LSD (Line Segment Detector)

Bindings over a modified parallelized version of the original C implementation of LSD, that allows to change the different thresholds involved and to provide custom image gradientes in order to compute the method with stronger features.

![](resources/example.jpg)

## Install
The current instructions were tested under Ubuntu 22.04:

```
sudo apt-get install build-essential cmake libopencv-dev
git clone --recursive https://github.com/francois141/faster-pylsd.git
cd faster-pylsd
pip3 install -r requirements.txt
pip3 install .
```

## Execution

```
python3 tests/test.py
```

