import ctypes
import numpy as np
from pathlib import Path
import time
import os

# 컴파일된 라이브러리 경로
project_root = Path(__file__).parent.parent
build_path = project_root / "build"

# 대체 경로 확인
lib_path = build_path / "kernel.so"
lib_cpu_path = build_path / "kernel_cpu.so"

# 경로 디버깅
if not lib_path.exists():
    print(f"⚠️  경고: {lib_path} 을(를) 찾을 수 없습니다")
if not lib_cpu_path.exists():
    print(f"⚠️  경고: {lib_cpu_path} 을(를) 찾을 수 없습니다")

cuda_available = False
cpu_cpp_available = False

try:
    lib = ctypes.CDLL(str(lib_path))
    cuda_available = True
    cuda_vector_add = lib.cuda_vector_add
    cuda_vector_add.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    cuda_vector_add.restype = None
    print("✓ CUDA 라이브러리 로드됨")
except OSError as e:
    print(f"⚠️  CUDA 라이브러리 로드 실패: {e}")

# CPU C++ 함수 로드 시도
try:
    lib_cpu = ctypes.CDLL(str(lib_cpu_path))
    cpu_cpp_available = True
    cpu_cpp_vector_add = lib_cpu.cpu_vector_add
    cpu_cpp_vector_add.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    cpu_cpp_vector_add.restype = None
    print("✓ CPU C++ 라이브러리 로드됨")
except OSError as e:
    print(f"⚠️  CPU C++ 라이브러리 로드 실패: {e}")

# 1. CPU Python
def vector_add_cpu_python(a, b):
    """순수 Python + numpy로 처리"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    
    if a.shape != b.shape:
        raise ValueError("입력 배열의 크기가 다릅니다")
    
    return a + b

# 2. CPU C++
def vector_add_cpu_cpp(a, b):
    """C++로 컴파일된 CPU 함수 호출"""
    if not cpu_cpp_available:
        raise RuntimeError("CPU C++ 라이브러리를 사용할 수 없습니다")
    
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    
    if a.shape != b.shape:
        raise ValueError("입력 배열의 크기가 다릅니다")
    
    N = len(a)
    c = np.zeros(N, dtype=np.float32)
    
    cpu_cpp_vector_add(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        N
    )
    
    return c

# 3. CUDA C++ (전체 오버헤드 포함)
def vector_add_cuda_cpp(a, b):
    """CUDA로 컴파일된 함수 호출 (메모리 생성/복사/계산/복사 모두 포함)"""
    if not cuda_available:
        raise RuntimeError("CUDA 라이브러리를 사용할 수 없습니다")
    
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    
    if a.shape != b.shape:
        raise ValueError("입력 배열의 크기가 다릅니다")
    
    N = len(a)
    c = np.zeros(N, dtype=np.float32)
    
    cuda_vector_add(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        N
    )
    
    return c

# 4. Numpy
def vector_add_numpy(a, b):
    """numpy 벡터화 연산"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    
    if a.shape != b.shape:
        raise ValueError("입력 배열의 크기가 다릅니다")
    
    return np.add(a, b)

def benchmark_method(name, func, a, b, iterations=3):
    """벤치마크 실행"""
    try:
        times = []
        
        # 워밍업
        func(a, b)
        
        # 반복 측정
        for _ in range(iterations):
            start = time.perf_counter()
            result = func(a, b)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"✓ {name:15} | {avg_time*1000:8.3f} ms ± {std_time*1000:6.3f} ms")
        return result, avg_time
        
    except Exception as e:
        print(f"✗ {name:15} | 오류: {str(e)}")
        return None, float('inf')

def run_benchmark(sizes=[1_000_000, 10_000_000, 100_000_000]):
    """모든 벤치마크 실행 및 결과 반환"""
    methods = ['CPU Python', 'CPU C++', 'CUDA C++', 'Numpy']
    all_results = {}
    
    for size in sizes:
        all_results[size] = {'times': {}, 'errors': {}}
    
    print("\n" + "="*80)
    print("📌 주의: CUDA는 메모리 할당/복사/계산/복사 전체 오버헤드를 포함합니다")
    print("   단순 덧셈 같은 작은 작업에서는 CPU가 더 빠를 수 있습니다.")
    print("="*80)
    
    for N in sizes:
        print(f"\n데이터 크기: {N:,} 개 ({N*4/1024/1024:.1f} MB)")
        print("-"*80)
        
        # 테스트 데이터 생성
        a = np.random.rand(N).astype(np.float32)
        b = np.random.rand(N).astype(np.float32)
        
        results = {}
        times = {}
        
        # 1. CPU Python
        result, elapsed = benchmark_method("1. CPU Python", vector_add_cpu_python, a, b)
        results['CPU Python'] = result
        times['CPU Python'] = elapsed
        
        # 2. CPU C++
        result, elapsed = benchmark_method("2. CPU C++", vector_add_cpu_cpp, a, b)
        results['CPU C++'] = result
        times['CPU C++'] = elapsed
        
        # 3. CUDA C++
        result, elapsed = benchmark_method("3. CUDA C++", vector_add_cuda_cpp, a, b)
        results['CUDA C++'] = result
        times['CUDA C++'] = elapsed
        
        # 4. Numpy
        result, elapsed = benchmark_method("4. Numpy", vector_add_numpy, a, b)
        results['Numpy'] = result
        times['Numpy'] = elapsed
        
        # 결과 검증
        print("-"*80)
        print("결과 검증:")
        reference = results['CPU Python']
        for method_name, result in results.items():
            if result is not None:
                error = np.max(np.abs(result - reference))
                print(f"  {method_name:15} | 최대 오차: {error:.2e}")
                all_results[N]['errors'][method_name] = error
                all_results[N]['times'][method_name] = times[method_name]
        
        # 속도 비교 (CPU Python 기준)
        print("-"*80)
        print("상대 속도 (CPU Python = 1.0배):")
        baseline = times['CPU Python']
        for method_name, elapsed in sorted(times.items(), key=lambda x: x[1]):
            if elapsed != float('inf'):
                ratio = baseline / elapsed
                symbol = "🚀" if ratio > 1 else "🐢"
                print(f"  {symbol} {method_name:15} | {ratio:6.2f}배")
    
    return all_results, sizes, methods