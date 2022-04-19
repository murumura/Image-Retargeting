#include <image/compute_saliency.h>
#include <iostream>
#include <stdio.h>
namespace Image {

    namespace cudaUtils {

        __host__ __device__ inline float sigmoidCuda(const float x, const float alpha, const float beta = 0.f)
        {
            const float val = (x - beta) / alpha;
            return 1.0f / (1.0f + expf(-val));
        }

        template <typename val_t>
        class MinK {
        public:
            // Constructor.
            //
            // Arguments:
            //   vals: Color distance to key track of
            //   K: How many values to keep track of
            __device__ MinK(val_t* vals, int K)
                : vals(vals), K(K), _size(0) {}

            // Try to add a new key and associated value to the data structure. If the key
            // is one of the smallest K seen so far then it will be kept; otherwise it
            // it will not be kept.
            // Arguments:
            //   val: The value associated to the key
            __device__ __forceinline__ void add(const val_t& val)
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
                        val_t cur_val = vals[k];
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
                            val_t val = vals[j];
                            vals[j] = vals[j + 1];
                            vals[j + 1] = val;
                        }
                    }
                }
            }

        private:
            val_t* vals;
            int K;
            int _size;
            val_t max_val;
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
        const float* __restrict__ singleScalePatch,
        const float* __restrict__ multiScalePatch,
        const int* __restrict__ indices,
        float* __restrict__ output,
        const int distC, const int K,
        const int B, const int pH, const int pW, const int C,
        const int H, const int W)
    {
        int index = threadIdx.x + (blockDim.x * blockIdx.x);

        float diffValues[MAXK];

        float minColorDist = 2e5, maxColorDist = 2e-5;
        // maintain k-smallest elements
        cudaUtils::MinK<float> mink(diffValues, K);
        int calcR = index / pW;
        int calcC = index % pW;
        if (calcR >= pH || calcC >= pW)
            return;

        float colorDist = 0;
        float posDist = 0;
        const int L = max(H, W);

        Index pixel_i = calcR * pW * 2 + calcC * 2;
        const int pixelR = indices[pixel_i];
        const int pixelC = indices[pixel_i + 1];

        for (int batch = 0; batch < B; batch++)
            for (int p_row = 0; p_row < pH; p_row++)
                for (int p_col = 0; p_col < pW; p_col++) {
                    if (p_row == calcR && p_col == calcC)
                        continue;
                    colorDist = 0;
                    for (int ch = 0; ch < C; ch++) {
                        Index i1 = calcR * pW * C + calcC * C + ch;
                        Index i2 = batch * pH * pW * C + p_row * pW * C + p_col * C + ch;
                        colorDist += powf((singleScalePatch[i1] - multiScalePatch[i2] + 0.0), 2);
                    }
                    colorDist = cudaUtils::sigmoidCuda(sqrt(colorDist), 0.1, 0.0);
                    Index i = p_row * pW * 2 + p_col * 2;
                    const int row = indices[i];
                    const int col = indices[i + 1];
                    float dRow = (pixelR - row + 1e-3);
                    float dCol = (pixelC - col + 1e-3);
                    posDist = sqrt(dRow * dRow + dCol * dCol) / L;
                    float dist = colorDist / (1.0 + distC * posDist);
                    mink.add(dist);
                    if (colorDist > maxColorDist)
                        maxColorDist = colorDist;
                    if (colorDist < minColorDist)
                        minColorDist = colorDist;
                }
        mink.sort();

        float sum = 0;
        int n = 0;
        for (n = 0; n < K && n < mink.size(); n++)
            sum += diffValues[n];

        output[index] = 1 - expf(-sum / n + 0.0);
    }

    void calcSaliencyValueCuda(
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& singleScalePatch,
        const Eigen::Tensor<float, 4, Eigen::RowMajor>& multiScalePatches,
        const Eigen::Tensor<int, 3, Eigen::RowMajor>& indices,
        Eigen::Tensor<float, 3, Eigen::RowMajor>& salienceMap,
        const int distC,
        const int K, const int H, const int W)
    {
        const int B = multiScalePatches.dimension(0);
        const int pH = multiScalePatches.dimension(1);
        const int pW = multiScalePatches.dimension(2);
        const int C = multiScalePatches.dimension(3);

        const int nThread = pW;
        const int nBlock = iDivUp(pH * pW, nThread);

        std::size_t multiScaleTensorBytes = multiScalePatches.size() * sizeof(float);
        std::size_t singleScaleTensorBytes = singleScalePatch.size() * sizeof(float);
        std::size_t indicesBytes = indices.size() * sizeof(int);
        std::size_t outTensorBytes = salienceMap.size() * sizeof(float);

        float* singleScaleDevice;
        float* multiScaleDevice;
        int* indicesDevice;
        float* imgOutDevice;
        // Allocate device memory
        CUDA_CHECK(cudaMalloc((void**)(&singleScaleDevice), singleScaleTensorBytes));
        CUDA_CHECK(cudaMalloc((void**)(&multiScaleDevice), multiScaleTensorBytes));
        CUDA_CHECK(cudaMalloc((void**)(&indicesDevice), indicesBytes));
        CUDA_CHECK(cudaMalloc((void**)(&imgOutDevice), outTensorBytes));

        // Copy data from host to device
        CUDA_CHECK(cudaMemcpy(singleScaleDevice, singleScalePatch.data(), singleScaleTensorBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(multiScaleDevice, multiScalePatches.data(), multiScaleTensorBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(indicesDevice, indices.data(), indicesBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(imgOutDevice, 0.0, outTensorBytes));

        // Launch kernel
        calcSaliencyCudaKernel<<<nBlock, nThread>>>(singleScaleDevice, multiScaleDevice, indicesDevice, imgOutDevice, distC, K, B, pH, pW, C, H, W);
        CUDA_CHECK(cudaGetLastError());

        // Blocks until the device has completed all preceding requested tasks
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(salienceMap.data(), imgOutDevice, outTensorBytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(singleScaleDevice));
        CUDA_CHECK(cudaFree(multiScaleDevice));
        CUDA_CHECK(cudaFree(imgOutDevice));
        CUDA_CHECK(cudaGetLastError());
    }
} // namespace Image
