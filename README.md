# CUDA Logistic Regression
## Requirements
- **CUDA Toolkit**
- **cnpy library** for reading `.npy` files  
## Project Structure
```
.
├── cuda_logreg.cu # CUDA implementation of Logistic Regression
├── benchmark.cpp # Benchmark script using .npy datasets
├── cnpy.h / cnpy.cpp # Library for reading .npy files
├── datasets/ # Example datasets in NumPy format
│ ├── X_100k_1k.npy
│ ├── y_100k_1k.npy
│ └── ...
├── README.md # This file
└── LICENSE
