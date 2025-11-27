# CUDA Logistic Regression
## Requirements
- **CUDA Toolkit**
- **nvcc**
- **cnpy library** for reading `.npy` files
- **vcpkg**
## Project Structure
```
.
├── cuda_logreg.cu 
├── benchmark.cpp
├── cnpy.h / cnpy.cpp
├── datasets/
│ ├── X_100k_1k.npy
│ ├── y_100k_1k.npy
│ └── ...
├── vcpkg
└── README.md
```
## Usage
Compile with nvcc on Windows CMD
```cmd
nvcc -allow-unsupported-compiler ^
    -I ./vcpkg/installed/x64-windows/include ^
    benchmark_cuda.cpp cnpy/cnpy.cpp cuda_logreg.cu ^
    -L ./vcpkg/installed/x64-windows/lib ^
    zlib.lib ^
    -o benchmark.exe
```
