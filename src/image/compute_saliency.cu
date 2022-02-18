#include <image/compute_saliency.h>
#include <iostream>
#include <stdio.h>
namespace Image {

    namespace cudaUtils {

        template <typename value_t>
        class MinK {
        public:
            // Constructor.
            //
            // Arguments:
            //   values: Array in which to store values
            //   K: How many values to keep track of
            __device__ MinK(value_t* vals, int K)
                : vals(vals), K(K), _size(0) {}

            // Try to add a new key and associated value to the data structure. If the key
            // is one of the smallest K seen so far then it will be kept; otherwise it
            // it will not be kept.
            //
            // Arguments:
            //   val: The value associated to the key
            __device__ __forceinline__ void add(const value_t& val)
            {
                if (_size < K) {
                    vals[_size] = val;
                    if (_size == 0 || val > max_val) {
                        max_val = val;
                        max_idx = _size;
                    }
                    _size++;
                }
                else if (val < max_val) {
                    vals[max_idx] = val;
                    max_val = val;
                    for (int k = 0; k < K; ++k) {
                        key_t cur_val = vals[k];
                        if (cur_val > max_val) {
                            max_val = cur_val;
                            max_idx = k;
                        }
                    }
                }
            }

            // Get the number of items currently stored in the structure.
            __device__ __forceinline__ int size()
            {
                return _size;
            }

            // Sort the items stored in the structure using bubble sort.
            __device__ __forceinline__ void sort()
            {
                for (int i = 0; i < _size - 1; ++i) {
                    for (int j = 0; j < _size - i - 1; ++j) {
                        if (vals[j + 1] < vals[j]) {
                            value_t val = vals[j];
                            vals[j] = vals[j + 1];
                            vals[j + 1] = val;
                        }
                    }
                }
            }

        private:
            value_t* vals;
            int K;
            int _size;
            value_t max_val;
            int max_idx;
        };

        void check(cudaError_t err, const char* prefix, const char* file, int line)
        {
            if (err != cudaSuccess) {
                std::ostringstream ess;
                ess << prefix << '[' << file << ':' << line
                    << "] CUDA error: " << cudaGetErrorString(err);
                throw std::runtime_error(ess.str());
            }
        }

        void check(cudaError_t err, const char* file, int line)
        {
            check(err, "", file, line);
        }
    } // namespace cudaUtils

/// usage: `CUDA_CHECK(cudaError_t err[, const char* prefix])`
#define CUDA_CHECK(...) \
    cudaUtils::check(__VA_ARGS__, __FILE__, __LINE__)

#define MAXK 1000

    // Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

    __global__ void calcSaliencyCudaKernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        const int distC, const int K,
        const int H, const int W, const int C)
    {
        int index = threadIdx.x + (blockDim.x * blockIdx.x);
        float diffValues[MAXK];
        // maintain k-smallest elements
        cudaUtils::MinK<float> mink(diffValues, K);
        int calcR = index / W;
        int calcC = index % W;
        if (calcR >= H || calcC >= W)
            return;
        float colorDist = 0;
        float posDist = 0;
        const int L = max(H, W);
        for (int row = 0; row < H; row++)
            for (int col = 0; col < W; col++) {
                colorDist = 0;
                for (int ch = 0; ch < C; ch++) {
                    Index i1 = calcR * W * C + calcC * C + ch;
                    Index i2 = row * W * C + col * C + ch;
                    colorDist += powf((input[i1] - input[i2] + 0.0), 2);
                }
                colorDist = sqrt(colorDist);
                float dRow = (calcR - row + 0.0) / L;
                float dCol = (calcC - col + 0.0) / L;
                posDist = sqrt(dRow * dRow + dCol * dCol);
                float dist = colorDist / (1.0 + distC * posDist);
                mink.add(dist);
            }
        mink.sort();
        float sum = 0;
        int n = 0;
        for (n = 0; n < K && n < mink.size(); n++)
            sum += diffValues[n];

        output[index] = 1 - expf(-sum / n);
    }

    void calcSaliencyValueCuda(
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& imgSrcLAB,
        Eigen::Tensor<float, 3, Eigen::RowMajor>& salienceMap,
        int distC,
        int K)
    {
        const int H = imgSrcLAB.dimension(0);
        const int W = imgSrcLAB.dimension(1);
        const int C = imgSrcLAB.dimension(2);

        const int nThread = W;
        const int nBlock = iDivUp(H * W, nThread);

        std::size_t inTensorBytes = imgSrcLAB.size() * sizeof(float);
        std::size_t outTensorBytes = salienceMap.size() * sizeof(float);
        float* imgSrcDevice;
        float* imgOutDevice;

        // Allocate device memory
        CUDA_CHECK(cudaMalloc((void**)(&imgSrcDevice), inTensorBytes));
        CUDA_CHECK(cudaMalloc((void**)(&imgOutDevice), outTensorBytes));

        // Copy data from host to device
        CUDA_CHECK(cudaMemcpy(imgSrcDevice, imgSrcLAB.data(), inTensorBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(imgOutDevice, 0.0, outTensorBytes));
        // Launch kernel
        calcSaliencyCudaKernel<<<nBlock, nThread>>>(imgSrcDevice, imgOutDevice, distC, K, H, W, C);
        CUDA_CHECK(cudaGetLastError());

        // Blocks until the device has completed all preceding requested tasks
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(salienceMap.data(), imgOutDevice, outTensorBytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(imgSrcDevice));
        CUDA_CHECK(cudaFree(imgOutDevice));
        CUDA_CHECK(cudaGetLastError());
    }
} // namespace Image
