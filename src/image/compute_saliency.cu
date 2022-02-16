#include <image/compute_saliency.h>
#include <stdio.h>
#include <iostream>
namespace Image {

    namespace cudaUtils {
        // A data structure to keep track of the smallest K keys seen so far as well
        // as their associated values, intended to be used in device code.
        // This data structure doesn't allocate any memory; keys and values are stored
        // in arrays passed to the constructor.
        //
        // The implementation is generic; it can be used for any key type that supports
        // the < operator, and can be used with any value type.
        //
        // Example usage:
        //
        // float keys[K];
        // int values[K];
        // MinK<float, int> mink(keys, values, K);
        // for (...) {
        //   // Produce some key and value from somewhere
        //   mink.add(key, value);
        // }
        // mink.sort();
        //
        // Now keys and values store the smallest K keys seen so far and the values
        // associated to these keys:
        //
        // for (int k = 0; k < K; ++k) {
        //   float key_k = keys[k];
        //   int value_k = values[k];
        // }
        template <typename key_t, typename value_t>
        class MinK {
        public:
            // Constructor.
            //
            // Arguments:
            //   keys: Array in which to store keys
            //   values: Array in which to store values
            //   K: How many values to keep track of
            __device__ MinK(key_t* keys, value_t* vals, int K)
                : keys(keys), vals(vals), K(K), _size(0) {}

            // Try to add a new key and associated value to the data structure. If the key
            // is one of the smallest K seen so far then it will be kept; otherwise it
            // it will not be kept.
            //
            // This takes O(1) operations if the new key is not kept, or if the structure
            // currently contains fewer than K elements. Otherwise this takes O(K) time.
            //
            // Arguments:
            //   key: The key to add
            //   val: The value associated to the key
            __device__ __forceinline__ void add(const key_t& key, const value_t& val)
            {
                if (_size < K) {
                    keys[_size] = key;
                    vals[_size] = val;
                    if (_size == 0 || key > max_key) {
                        max_key = key;
                        max_idx = _size;
                    }
                    _size++;
                }
                else if (key < max_key) {
                    keys[max_idx] = key;
                    vals[max_idx] = val;
                    max_key = key;
                    for (int k = 0; k < K; ++k) {
                        key_t cur_key = keys[k];
                        if (cur_key > max_key) {
                            max_key = cur_key;
                            max_idx = k;
                        }
                    }
                }
            }

            // Get the number of items currently stored in the structure.
            // This takes O(1) time.
            __device__ __forceinline__ int size()
            {
                return _size;
            }

            // Sort the items stored in the structure using bubble sort.
            // This takes O(K^2) time.
            __device__ __forceinline__ void sort()
            {
                for (int i = 0; i < _size - 1; ++i) {
                    for (int j = 0; j < _size - i - 1; ++j) {
                        if (keys[j + 1] < keys[j]) {
                            key_t key = keys[j];
                            value_t val = vals[j];
                            keys[j] = keys[j + 1];
                            vals[j] = vals[j + 1];
                            keys[j + 1] = key;
                            vals[j + 1] = val;
                        }
                    }
                }
            }

        private:
            key_t* keys;
            value_t* vals;
            int K;
            int _size;
            key_t max_key;
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

#define MAXK 1000 * 1000
#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8

    // Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

    __global__ void calcSaliencyCudaKernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        const int distC, int K,
        const int H, const int W, const int C)
    {
        int index = threadIdx.x + (blockDim.x * blockIdx.x);
        float diffValues[MAXK];
        int keys[MAXK];
        // maintain k-smallest elements
        cudaUtils::MinK<int, float> mink(keys, diffValues, K);
        int calcR = index / W;
        int calcC = index % W;
        if(calcR >= H || calcC >= W) 
            return;
        float colorDist = 0;
        float posDist = 0;

        for (int row = 0; row < H; row++)
            for (int col = 0; col < W; col++) 
            {
                colorDist = 0;
                for (int ch = 0; ch < C; ch++) 
                {
                    int i1 = calcR * W * C + calcC * C + ch;
                    int i2 = row * W * C + col * C + ch;
                    colorDist += (input[i1] - input[i2]) * (input[i1] - input[i2]);
                }
                colorDist = sqrtf(colorDist);
                float dRow = (calcR - row + 0.0) / H;
                float dCol = (calcC - col + 0.0) / W;
                posDist = sqrtf(dRow * dRow + dCol * dCol);
                float dist = colorDist / (1 + distC * posDist);
                mink.add(row * W + col, dist);
            }

        mink.sort();
        float sum = 0;
        int n = 0;
        for (n = 0; n <= K && n < mink.size(); n++)
            sum += diffValues[n];

        output[calcR * W + calcC] = 1 - exp(-sum / n);
    }

    void calcSaliencyValueCuda(
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& imgSrcLAB,
        Eigen::Tensor<float, 3, Eigen::RowMajor>& salienceMap,
        int distC, int K)
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
