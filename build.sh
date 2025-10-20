#!/bin/bash

echo "================================================================================"
echo "🔨 빌드 시작..."
echo "================================================================================"

# 프로젝트 루트 디렉토리
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
SRC_DIR="$PROJECT_ROOT/src"

# build 디렉토리 생성
mkdir -p "$BUILD_DIR"

cd "$SRC_DIR" || exit

echo ""
echo "📁 소스 디렉토리: $SRC_DIR"
echo "📁 빌드 디렉토리: $BUILD_DIR"
echo ""

# CPU C++ 컴파일 (최적화)
echo "🔧 CPU C++ 컴파일 중..."
if g++ -shared -fPIC -O3 -march=native -fopenmp -o "$BUILD_DIR/kernel_cpu.so" kernel_cpu.cpp; then
    echo "✓ CPU C++ 컴파일 성공"
    echo "   옵션: -O3 -march=native -fopenmp"
else
    echo "✗ CPU C++ 컴파일 실패"
    exit 1
fi

echo ""

# CUDA C++ 컴파일 (최적화)
echo "🔧 CUDA C++ 컴파일 중..."
if nvcc -shared -Xcompiler -fPIC -O3 -arch=sm_75 -o "$BUILD_DIR/kernel.so" kernel.cu 2>/dev/null; then
    echo "✓ CUDA C++ 컴파일 성공"
    echo "   옵션: -O3 -arch=sm_75"
elif nvcc -shared -Xcompiler -fPIC -O3 -o "$BUILD_DIR/kernel.so" kernel.cu; then
    echo "✓ CUDA C++ 컴파일 성공 (자동 아키텍처 선택)"
    echo "   옵션: -O3"
else
    echo "⚠️  CUDA 컴파일 실패 - CPU 전용으로 계속 진행"
fi

echo ""
echo "================================================================================"
echo "✓ 빌드 완료!"
echo "================================================================================"
echo ""
echo "📊 생성된 파일:"
ls -lh "$BUILD_DIR"
echo ""
echo "🚀 실행 방법: python $PROJECT_ROOT/main.py"