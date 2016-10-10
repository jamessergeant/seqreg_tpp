
// #ifndef _SEQSLAM_KERNEL_CH_
// #define _SEQSLAM_KERNEL_CH_
//
// #include <helper_functions.h>
// #include <helper_math.h>
//
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
//
// #include <math.h>
// #include <string>
// #include <typeinfo>
// #include <vector>


////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host
// call returns an error
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
                (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void local_norm(const float *id, float *od, int w, int h, int r,
                              float minstd) {
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if ((xIndex >= h) || (yIndex >= w)) return;

    float mean = 0.0f;

    int N = 0;

    int x_min = max(xIndex - r, 0);
    int x_max = min(xIndex + r + 1, h);
    int y_min = max(yIndex - r, 0);
    int y_max = min(yIndex + r + 1, w);

    for (int x = x_min; x < x_max; x++) {
        for (int y = y_min; y < y_max; y++) {
            mean += id[y * h + x];
            N = N + 1;
        }
    }

    mean /= static_cast<float>(N);

    float sd = 0.0f;

    for (int x = x_min; x < x_max; x++) {
        for (int y = y_min; y < y_max; y++) {
            float pVal = id[y * h + x];
            pVal -= mean;
            sd += pow(pVal, 2);
        }
    }

    sd /= static_cast<float>(N-1);

    sd = sqrt(sd);

    if (sd < minstd) {
        od[yIndex * h + xIndex] = 0.0;
    } else {
        od[yIndex * h + xIndex] = (id[yIndex * h + xIndex] - mean) / sd;
    }

}
