#include <cuda_runtime.h>
#include <stdio.h>

// CUDA 에러 확인 매크로
#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA 커널: 벡터 더하기
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// C 인터페이스 - 최적화된 버전
extern "C" {
    // 장치 메모리에서 계산하고 결과를 반환
    void cuda_vector_add(const float* h_A, const float* h_B, float* h_C, int N) {
        float* d_A = nullptr;
        float* d_B = nullptr;
        float* d_C = nullptr;
        
        size_t bytes = N * sizeof(float);
        
        // 장치 메모리 할당
        cudaCheck(cudaMalloc(&d_A, bytes));
        cudaCheck(cudaMalloc(&d_B, bytes));
        cudaCheck(cudaMalloc(&d_C, bytes));
        
        // 호스트에서 장치로 데이터 복사 (비동기)
        cudaCheck(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice));
        
        // 커널 실행 - 최적화된 블록 크기
        int blockSize = 512;  // 더 큰 블록 크기
        int gridSize = (N + blockSize - 1) / blockSize;
        vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        cudaCheck(cudaGetLastError());
        
        // 장치에서 호스트로 결과 복사 (비동기)
        cudaCheck(cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        
        // 모든 작업 완료 대기
        cudaCheck(cudaDeviceSynchronize());
        
        // 메모리 해제
        cudaCheck(cudaFree(d_A));
        cudaCheck(cudaFree(d_B));
        cudaCheck(cudaFree(d_C));
    }
}