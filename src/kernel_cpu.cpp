#include <omp.h>

// CPU C++ 버전: 병렬 처리 (OpenMP 사용)
extern "C"
{
    void cpu_vector_add(const float *A, const float *B, float *C, int N)
    {
#pragma omp parallel for
        for (int i = 0; i < N; i++)
        {
            C[i] = A[i] + B[i];
        }
    }
}