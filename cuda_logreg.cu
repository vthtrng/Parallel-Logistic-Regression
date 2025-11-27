// -------------------------------------------------------------
// cuda_logreg.cu - Parallel Logistic Regression 
// -------------------------------------------------------------
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// -----------------------------------------------------------------------------
// GPU KERNELS
// -----------------------------------------------------------------------------

// Algorithm 2: vector-matrix mul (scale rows)
__global__ void vecMatMulKernel(float *M, const float *v, int m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    float scale = v[i];
    for (int j = 0; j < n; j++)
        M[i*n + j] *= scale;
}

// Algorithm 3: matrix column sum
__global__ void colSumKernel(const float *M, float *res, int m, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    float s = 0.0f;
    for (int i = 0; i < m; i++)
        s += M[i*n + j];

    res[j] = s;
}

// Algorithm 5: subtract
__global__ void subtractKernel(const float *v1, float *res2, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    res2[i] -= v1[i];
}

// Algorithm 6: sigmoid
__global__ void sigmoidKernel(const float *res, float *ypred, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    ypred[i] = 1.0f / (1.0f + expf(-res[i]));
}

// Algorithm 4: norm2
__global__ void norm2Kernel(const float *v, float *out, int n)
{
    __shared__ float cache[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float temp = 0.0f;
    if (idx < n) temp = v[idx] * v[idx];
    cache[tid] = temp;
    __syncthreads();

    // reduction
    int s = blockDim.x / 2;
    while (s > 0) {
        if (tid < s) cache[tid] += cache[tid + s];
        __syncthreads();
        s /= 2;
    }

    if (tid == 0) atomicAdd(out, cache[0]);
}

// -----------------------------------------------------------------------------
// HOST API – logistic_regression_train()
// -----------------------------------------------------------------------------
extern "C" void logistic_regression_train(
        float *X, float *Y, float *w,
        int m, int n,
        float alpha, float eps,
        int max_iter)
{
    // GPU memory
    float *dX, *dXT, *dY, *dw, *dRes, *dLoss;
    cudaMalloc(&dX, m*n*sizeof(float));
    cudaMalloc(&dXT, m*n*sizeof(float));
    cudaMalloc(&dY, m*sizeof(float));
    cudaMalloc(&dw, n*sizeof(float));
    cudaMalloc(&dRes, m*sizeof(float));
    cudaMalloc(&dLoss, n*sizeof(float));

    cudaMemcpy(dX, X, m*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w, n*sizeof(float), cudaMemcpyHostToDevice);

    // Precompute transpose of X
    float *hXT = (float *)malloc(m*n*sizeof(float));
    for (int i=0;i<m;i++)
        for (int j=0;j<n;j++)
            hXT[j*m + i] = X[i*n + j];
    cudaMemcpy(dXT, hXT, m*n*sizeof(float), cudaMemcpyHostToDevice);
    free(hXT);

    dim3 block(256);
    dim3 grid_m((m+255)/256);
    dim3 grid_n((n+255)/256);

    for (int it=0; it < max_iter; it++) {

        // res = w * X^T
        cudaMemset(dXT, 0, m*n*sizeof(float));
        vecMatMulKernel<<<grid_m,block>>>(dXT, dw, m, n);
        colSumKernel<<<grid_m,block>>>(dXT, dRes, m, n);

        // sigmoid(res)
        sigmoidKernel<<<grid_m,block>>>(dRes, dRes, m);

        // res = Y - res
        subtractKernel<<<grid_m,block>>>(dRes, (float*)dY, m);

        // compute gradient
        cudaMemset(dX, 0, m*n*sizeof(float));
        vecMatMulKernel<<<grid_m,block>>>(dX, dY, m, n);
        colSumKernel<<<grid_n,block>>>(dX, dLoss, m, n);

        // w -= alpha * grad
        float *hLoss = (float*)malloc(n*sizeof(float));
        cudaMemcpy(hLoss, dLoss, n*sizeof(float), cudaMemcpyDeviceToHost);
        for (int j=0;j<n;j++) hLoss[j] *= alpha;
        float *hW = (float*)malloc(n*sizeof(float));
        cudaMemcpy(hW, dw, n*sizeof(float), cudaMemcpyDeviceToHost);
        for (int j=0;j<n;j++) hW[j] -= hLoss[j];
        cudaMemcpy(dw, hW, n*sizeof(float), cudaMemcpyHostToDevice);
        free(hLoss);
        free(hW);

        // ||grad||²
        float zero = 0;
        float *dNorm;
        cudaMalloc(&dNorm, sizeof(float));
        cudaMemcpy(dNorm, &zero, sizeof(float), cudaMemcpyHostToDevice);

        norm2Kernel<<<grid_n,block>>>(dLoss, dNorm, n);

        float hNorm;
        cudaMemcpy(&hNorm, dNorm, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dNorm);

        if (sqrt(hNorm) < eps) break;
    }

    cudaMemcpy(w, dw, n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dX); cudaFree(dXT); cudaFree(dY);
    cudaFree(dw); cudaFree(dRes); cudaFree(dLoss);
}
