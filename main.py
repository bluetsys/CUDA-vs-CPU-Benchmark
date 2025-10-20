#!/usr/bin/env python3
"""
벡터 더하기 성능 비교 메인 프로그램
- modules/benchmark.py: 벤치마크 실행
- modules/chart.py: 차트 생성 및 결과 출력
"""

import sys
from pathlib import Path

# modules 폴더를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from benchmark import run_benchmark
from chart import create_performance_charts, print_summary

def main():
    print("\n" + "="*80)
    print("벡터 더하기 성능 비교: CPU Python vs CPU C++ vs CUDA C++ vs Numpy")
    print("="*80)
    
    # 벤치마크 실행
    # 참고: CUDA는 메모리 전송 오버헤드 때문에 단순 덧셈에서는 느립니다
    sizes = [100_000, 1_000_000, 10_000_000, 100_000_000]
    all_results, sizes, methods = run_benchmark(sizes)
    
    # 종합 결과 출력
    print_summary(all_results, sizes, methods)
    
    # 차트 생성
    print("\n📈 차트 생성 중...")
    create_performance_charts(all_results, sizes, methods)
    print("✓ 차트 저장 완료")

if __name__ == "__main__":
    main()