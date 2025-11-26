"""
Lightweight profiling utilities for use in Jupyter notebooks.
Import this and use the decorators/timers directly in your notebook cells.
"""

import time
import numpy as np
from functools import wraps
import sys


class DetailedTimer:
    """
    Hierarchical timer for profiling nested operations.
    
    Usage:
        timer = DetailedTimer("My Operation")
        with timer:
            with timer.section("Part 1"):
                # do work
                pass
            with timer.section("Part 2"):
                # do more work
                pass
        timer.print_report()
    """
    
    def __init__(self, name="Operation"):
        self.name = name
        self.sections = {}
        self.start_time = None
        self.total_time = None
        self.current_section = None
        
    def section(self, name):
        """Create a timed section"""
        return TimedSection(self, name)
    
    def record_section(self, name, duration):
        """Record time for a section"""
        if name not in self.sections:
            self.sections[name] = []
        self.sections[name].append(duration)
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.total_time = time.perf_counter() - self.start_time
    
    def print_report(self, file=sys.stdout):
        """Print detailed timing report"""
        if self.total_time is None:
            print("Timer not completed", file=file)
            return
        
        print(f"\n{'='*70}", file=file)
        print(f"Timing Report: {self.name}", file=file)
        print(f"{'='*70}", file=file)
        print(f"Total time: {self.total_time*1000:.2f} ms ({self.total_time:.3f} s)", file=file)
        print(f"-"*70, file=file)
        
        if not self.sections:
            print("No sections recorded", file=file)
            return
        
        # Calculate section statistics
        section_stats = {}
        for name, times in self.sections.items():
            section_stats[name] = {
                'total': sum(times),
                'count': len(times),
                'mean': np.mean(times),
                'std': np.std(times) if len(times) > 1 else 0,
                'min': min(times),
                'max': max(times)
            }
        
        # Sort by total time
        sorted_sections = sorted(section_stats.items(), key=lambda x: x[1]['total'], reverse=True)
        
        print(f"{'Section':<30} {'Total':<12} {'Count':<8} {'Mean':<12} {'% of Total':<12}", file=file)
        print(f"-"*70, file=file)
        
        accounted_time = 0
        for name, stats in sorted_sections:
            total_ms = stats['total'] * 1000
            mean_ms = stats['mean'] * 1000
            pct = (stats['total'] / self.total_time) * 100
            accounted_time += stats['total']
            
            print(f"{name:<30} {total_ms:>10.2f}ms {stats['count']:>6}x {mean_ms:>10.2f}ms {pct:>10.1f}%", file=file)
        
        # Show unaccounted time
        unaccounted = self.total_time - accounted_time
        if unaccounted > 0.001:  # More than 1ms
            pct = (unaccounted / self.total_time) * 100
            print(f"-"*70, file=file)
            print(f"{'<unaccounted overhead>':<30} {unaccounted*1000:>10.2f}ms {'':>6}  {'':>10}  {pct:>10.1f}%", file=file)
        
        print(f"{'='*70}", file=file)


class TimedSection:
    """Context manager for timing a section"""
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name
        self.start = None
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        duration = time.perf_counter() - self.start
        self.timer.record_section(self.name, duration)


def profile_function(func):
    """
    Decorator to profile a function's execution time.
    
    Usage:
        @profile_function
        def my_function():
            # do work
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"⏱️  {func.__name__}: {elapsed*1000:.2f}ms ({elapsed:.3f}s)")
        return result
    return wrapper


def compare_functions(functions, *args, num_runs=3, **kwargs):
    """
    Compare execution time of multiple functions with the same arguments.
    
    Args:
        functions: List of (name, function) tuples
        *args: Arguments to pass to each function
        num_runs: Number of times to run each function
        **kwargs: Keyword arguments to pass to each function
    
    Returns:
        Dictionary of results
    
    Usage:
        results = compare_functions([
            ("Sequential", read_sequential),
            ("Parallel", read_parallel)
        ], filename, num_runs=5)
    """
    print(f"\n{'='*70}")
    print(f"Function Comparison ({num_runs} runs each)")
    print(f"{'='*70}")
    
    results = {}
    
    for name, func in functions:
        print(f"\nTesting: {name}")
        times = []
        
        for i in range(num_runs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                print(f"  Run {i+1}: {elapsed*1000:.2f}ms")
            except Exception as e:
                print(f"  Run {i+1}: FAILED - {e}")
                times.append(float('inf'))
        
        valid_times = [t for t in times if t != float('inf')]
        if valid_times:
            results[name] = {
                'times': valid_times,
                'mean': np.mean(valid_times),
                'std': np.std(valid_times),
                'min': min(valid_times),
                'max': max(valid_times),
                'success': len(valid_times) == num_runs
            }
        else:
            results[name] = {
                'times': [],
                'mean': float('inf'),
                'std': 0,
                'min': float('inf'),
                'max': float('inf'),
                'success': False
            }
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"{'Function':<20} {'Mean':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15}")
    print(f"-"*70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'])
    
    for name, stats in sorted_results:
        if stats['success']:
            mean_ms = stats['mean'] * 1000
            std_ms = stats['std'] * 1000
            min_ms = stats['min'] * 1000
            max_ms = stats['max'] * 1000
            print(f"{name:<20} {mean_ms:>12.2f}ms {std_ms:>12.2f}ms {min_ms:>12.2f}ms {max_ms:>12.2f}ms")
        else:
            print(f"{name:<20} {'FAILED':>12}")
    
    # Show speedups
    if len(sorted_results) > 1 and sorted_results[0][1]['success']:
        baseline_name, baseline_stats = sorted_results[0]
        baseline_time = baseline_stats['mean']
        
        print(f"\nSpeedups relative to {baseline_name}:")
        for name, stats in sorted_results[1:]:
            if stats['success']:
                slowdown = stats['mean'] / baseline_time
                if slowdown > 1:
                    print(f"  {name}: {slowdown:.2f}x SLOWER")
                else:
                    print(f"  {name}: {1/slowdown:.2f}x faster")
    
    print(f"{'='*70}\n")
    
    return results


# Simple inline profiling example
def profile_mmap_operations():
    """
    Example of how to use DetailedTimer for profiling mmap operations.
    Copy this pattern into your notebook.
    """
    print("Example profiling code:")
    print("""
import mmap
from notebook_profiler import DetailedTimer

timer = DetailedTimer("Block Reading")

with timer:
    # Get metadata
    with timer.section("get_metadata"):
        header, forest, block_info = get_metadata(filename)
    
    block_offsets = block_info[2]
    
    # Open mmap
    with timer.section("open_mmap"):
        f = open(filename, 'rb')
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    
    results = []
    for offset in block_offsets:
        # Read ghost cells
        with timer.section("read_ghost"):
            ghostcells = np.frombuffer(mm, dtype='=i4', count=2*ndim, offset=offset).copy()
        
        # Read field data
        with timer.section("read_field"):
            arr = np.frombuffer(mm, dtype='=f8', count=count, offset=byte_offset).copy()
        
        # Reshape
        with timer.section("reshape"):
            arr = arr.reshape(shape).T
        
        results.append(arr)
    
    # Cleanup
    with timer.section("cleanup"):
        mm.close()
        f.close()

timer.print_report()
    """)


if __name__ == '__main__':
    # Show example usage
    print("Notebook Profiler - Example Usage")
    print("="*70)
    
    # Example 1: Simple timer
    print("\n1. Simple timing:")
    timer = DetailedTimer("Example Operation")
    with timer:
        with timer.section("setup"):
            time.sleep(0.1)
        
        for i in range(5):
            with timer.section("iteration"):
                time.sleep(0.02)
        
        with timer.section("cleanup"):
            time.sleep(0.05)
    
    timer.print_report()
    
    # Example 2: Function comparison
    print("\n2. Function comparison:")
    
    def slow_function(n):
        return sum(i**2 for i in range(n))
    
    def fast_function(n):
        return sum(range(n))
    
    compare_functions([
        ("Slow", slow_function),
        ("Fast", fast_function)
    ], 100000, num_runs=5)

