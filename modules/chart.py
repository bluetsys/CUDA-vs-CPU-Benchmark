import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib
import platform
import subprocess
matplotlib.use('Agg')

# 한글 폰트 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)

# matplotlib 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

def get_system_info():
    """시스템 정보 수집"""
    info = {}
    
    # CPU 정보
    try:
        if platform.system() == 'Linux':
            result = subprocess.run(['lscpu'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'Model name' in line:
                    info['cpu'] = line.split(':', 1)[1].strip()
                elif 'CPU(s)' in line and 'CPU(s)' == line.split(':')[0].strip():
                    info['cpu_cores'] = line.split(':', 1)[1].strip()
        elif platform.system() == 'Darwin':  # macOS
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            info['cpu'] = result.stdout.strip()
        else:  # Windows
            import wmi
            c = wmi.WMI()
            for processor in c.Win32_Processor():
                info['cpu'] = processor.Name
    except:
        info['cpu'] = platform.processor() or 'Unknown'
    
    # GPU 정보
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                               '--format=csv,noheader'], capture_output=True, text=True)
        gpu_info = result.stdout.strip().split(', ')
        if len(gpu_info) >= 2:
            info['gpu'] = gpu_info[0]
            info['gpu_memory'] = gpu_info[1]
        else:
            info['gpu'] = result.stdout.strip()
    except:
        info['gpu'] = 'No CUDA GPU detected'
    
    # OS 정보
    info['os'] = platform.platform()
    
    # Python 버전
    info['python'] = platform.python_version()
    
    return info

def create_performance_charts(all_results, sizes, methods):
    """성능 비교 차트 생성"""
    # 시스템 정보 수집
    sys_info = get_system_info()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 제목에 시스템 정보 추가
    title = 'Vector Addition Performance Comparison\n'
    title += f"CPU: {sys_info.get('cpu', 'Unknown')[:50]}\n"
    title += f"GPU: {sys_info.get('gpu', 'Unknown')[:50]}"
    if 'gpu_memory' in sys_info:
        title += f" ({sys_info.get('gpu_memory')})"
    
    fig.suptitle(title, fontsize=13, fontweight='bold')
    
    # 색상 정의
    colors = {'CPU Python': '#FF6B6B', 'CPU C++': '#4ECDC4', 'CUDA C++': '#45B7D1', 'Numpy': '#FFA07A'}
    
    # 1. 실행 시간 비교 (선 그래프)
    ax1 = axes[0, 0]
    size_labels = [f'{s//1_000_000}M' if s >= 1_000_000 else f'{s//1_000}K' for s in sizes]
    for method in methods:
        times = []
        for size in sizes:
            if method in all_results[size]['times']:
                t = all_results[size]['times'][method]
                if t != float('inf'):
                    times.append(t * 1000)
                else:
                    times.append(0)
            else:
                times.append(0)
        ax1.plot(size_labels, times, marker='o', linewidth=2.5, markersize=8, label=method, color=colors[method])
    ax1.set_xlabel('Data Size', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('Execution Time Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. 상대 속도 비교 (막대 그래프)
    ax2 = axes[0, 1]
    x = np.arange(len(sizes))
    width = 0.2
    for i, method in enumerate(methods):
        speedups = []
        for size in sizes:
            baseline = all_results[size]['times'].get('CPU Python', float('inf'))
            current = all_results[size]['times'].get(method, float('inf'))
            if baseline != float('inf') and current != float('inf'):
                speedups.append(baseline / current)
            else:
                speedups.append(0)
        ax2.bar(x + i*width, speedups, width, label=method, color=colors[method])
    ax2.set_xlabel('Data Size', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Speedup (x)', fontsize=11, fontweight='bold')
    ax2.set_title('Speedup vs CPU Python (Log Scale)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(size_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # 3. 평균 성능 순위 (막대 그래프)
    ax3 = axes[1, 0]
    avg_times = {}
    for method in methods:
        valid_times = []
        for size in sizes:
            if method in all_results[size]['times']:
                t = all_results[size]['times'][method]
                if t != float('inf'):
                    valid_times.append(t)
        if valid_times:
            avg_times[method] = np.mean(valid_times)
    
    sorted_methods = sorted(avg_times.items(), key=lambda x: x[1])
    methods_sorted = [m[0] for m in sorted_methods]
    times_sorted = [m[1] * 1000 for m in sorted_methods]
    
    bars = ax3.barh(methods_sorted, times_sorted, color=[colors[m] for m in methods_sorted])
    ax3.set_xlabel('Average Execution Time (ms)', fontsize=11, fontweight='bold')
    ax3.set_title('Average Performance Ranking', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 값 표시
    for i, (method, time) in enumerate(zip(methods_sorted, times_sorted)):
        ax3.text(time, i, f' {time:.2f} ms', va='center', fontweight='bold')
    
    # 4. 데이터 크기별 모든 방법 비교 (그룹 막대)
    ax4 = axes[1, 1]
    x = np.arange(len(methods))
    width = 0.25
    for i, size in enumerate(sizes):
        times = []
        for method in methods:
            if method in all_results[size]['times']:
                t = all_results[size]['times'][method]
                if t != float('inf'):
                    times.append(t * 1000)
                else:
                    times.append(0)
            else:
                times.append(0)
        ax4.bar(x + i*width, times, width, label=f'{size//1_000_000}M' if size >= 1_000_000 else f'{size//1_000}K', alpha=0.8)
    ax4.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='bold')
    ax4.set_title('Execution Time by Data Size (Log Scale)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(methods, rotation=15, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_yscale('log')
    
    # 시스템 정보 추가
    system_text = f"System: {sys_info.get('os', 'Unknown')}\n"
    system_text += f"Python: {sys_info.get('python', 'Unknown')}"
    fig.text(0.99, 0.01, system_text, ha='right', va='bottom', fontsize=8, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ performance_comparison.png saved")
    print(f"\n📊 시스템 정보:")
    print(f"   CPU: {sys_info.get('cpu', 'Unknown')}")
    print(f"   GPU: {sys_info.get('gpu', 'Unknown')}")
    if 'gpu_memory' in sys_info:
        print(f"   GPU Memory: {sys_info.get('gpu_memory')}")
    print(f"   OS: {sys_info.get('os', 'Unknown')}")
    print(f"   Python: {sys_info.get('python', 'Unknown')}")
    plt.close()

def print_summary(all_results, sizes, methods):
    """종합 결과 출력"""
    print("\n" + "="*80)
    print("📊 종합 성능 비교 결과")
    print("="*80)
    
    # 실행 시간 비교표
    print("\n⏱️  실행 시간 (ms)")
    print("-"*80)
    size_labels = [f"{s//1000}K" if s < 1_000_000 else f"{s//1_000_000}M" for s in sizes]
    print(f"{'방법':<15} | {' | '.join(f'{label:>12}' for label in size_labels)}")
    print("-"*80)
    for method in methods:
        times_row = []
        for size in sizes:
            if method in all_results[size]['times']:
                t = all_results[size]['times'][method]
                if t != float('inf'):
                    times_row.append(f"{t*1000:8.3f} ms")
                else:
                    times_row.append("실패")
            else:
                times_row.append("실패")
        print(f"{method:<15} | {' | '.join(f'{t:>12}' for t in times_row)}")
    
    # 상대 속도 비교표 (CPU Python 기준)
    print("\n⚡ 상대 속도 (CPU Python = 1.0배)")
    print("-"*80)
    print(f"{'방법':<15} | {' | '.join(f'{label:>12}' for label in size_labels)}")
    print("-"*80)
    for method in methods:
        speed_row = []
        for size in sizes:
            baseline = all_results[size]['times'].get('CPU Python', float('inf'))
            current = all_results[size]['times'].get(method, float('inf'))
            if baseline != float('inf') and current != float('inf'):
                ratio = baseline / current
                symbol = "🚀" if ratio > 1 else "🐢"
                speed_row.append(f"{symbol} {ratio:6.2f}배")
            else:
                speed_row.append("N/A")
        print(f"{method:<15} | {' | '.join(f'{t:>12}' for t in speed_row)}")
    
    # 순위 표시
    print("\n🏆 성능 순위 (평균 실행 시간 기준)")
    print("-"*80)
    avg_times = {}
    for method in methods:
        valid_times = []
        for size in sizes:
            if method in all_results[size]['times']:
                t = all_results[size]['times'][method]
                if t != float('inf'):
                    valid_times.append(t)
        if valid_times:
            avg_times[method] = np.mean(valid_times)
    
    for rank, (method, avg_time) in enumerate(sorted(avg_times.items(), key=lambda x: x[1]), 1):
        print(f"{rank}. {method:<15} | 평균: {avg_time*1000:8.3f} ms")
    
    # CUDA 오버헤드 분석
    print("\n💡 CUDA 오버헤드 분석")
    print("-"*80)
    print("CUDA 시간 구성:")
    print("  1. GPU 메모리 할당        : ~1-5ms")
    print("  2. Host → Device 복사    : 데이터크기 / PCIe 대역폭")
    print("  3. GPU 커널 실행         : ~0.1-1ms (간단한 덧셈)")
    print("  4. Device → Host 복사    : 데이터크기 / PCIe 대역폭")
    print("  5. 동기화 및 정리        : ~0.1-1ms")
    print("")
    print("  PCIe 3.0 이론 대역폭 : ~4GB/s")
    print("  100MB 데이터 복사    : ~25ms (왕복 50ms)")
    print("")
    print("결론: 단순 덧셈은 메모리 전송 오버헤드 > 계산 시간")
    print("      복잡한 연산(행렬곱, AI)에서 CUDA가 유리합니다")
    
    print("\n" + "="*80)