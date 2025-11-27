// -------------------------------------------------------------
// benchmark.cpp
// -------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "cnpy.h"

extern "C" void logistic_regression_train(
        float *X, float *Y, float *w,
        int m, int n,
        float alpha, float eps,
        int max_iter);

// compute F1
float f1_score(const std::vector<float>& y, const std::vector<float>& p)
{
    int TP=0,FP=0,FN=0;
    for (int i=0;i<y.size();i++) {
        int yhat = p[i] >= 0.5f;
        if (y[i]==1 && yhat==1) TP++;
        else if (y[i]==0 && yhat==1) FP++;
        else if (y[i]==1 && yhat==0) FN++;
    }
    float f1 = TP ? 2.0f*TP/(2*TP + FP + FN) : 0;
    return f1;
}

int main() {
    std::ofstream log("benchmark_cuda_results.txt", std::ios::app);

    std::vector<std::pair<std::string,std::string>> datasets = {
        {"datasets/X_100k_1k.npy", "datasets/y_100k_1k.npy"},
        {"datasets/X_1m_100.npy",  "datasets/y_1m_100.npy"},
        {"datasets/X_HIGGS_6M.npy",  "datasets/y_HIGGS_6M.npy"},
        {"datasets/X_HIGGS_full.npy",  "datasets/y_HIGGS_full.npy"}
    };

    for (auto &ds : datasets)
    {
        cnpy::NpyArray Xnp = cnpy::npy_load(ds.first);
        cnpy::NpyArray Ynp = cnpy::npy_load(ds.second);

        float *X = Xnp.data<float>();
        float *Y = Ynp.data<float>();

        int m = Xnp.shape[0];
        int n = Xnp.shape[1];

        std::vector<float> w(n, 0.0f);

        auto start = std::chrono::high_resolution_clock::now();

        logistic_regression_train(
            X, Y, w.data(),
            m, n,
            0.01f, 1e-5f, 200
        );

        auto end = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration<double>(end-start).count();

        // predict
        std::vector<float> pred(m,0);
        for (int i=0;i<m;i++) {
            float z = 0;
            for (int j=0;j<n;j++) z += X[i*n + j]*w[j];
            pred[i] = 1.0/(1+exp(-z));
        }

        float f1 = f1_score(
            std::vector<float>(Y, Y+m), pred
        );

        log << ds.first << ", time=" << t << " sec, F1=" << f1 << "\n";

        std::cout << ds.first << " done.\n";
    }

    return 0;
}
