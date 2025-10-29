"""Concurrent execution benchmarking for LangGraph.

This module provides comprehensive testing of concurrent graph execution
patterns, including thread pools, process pools, and async concurrency.
"""

import asyncio
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import multiprocessing as mp
from collections import defaultdict

try:
    from langgraph.pregel import Pregel
except ImportError:
    # Mock class for testing
    class Pregel:
        pass


class ExecutionMode(Enum):
    """Different execution modes for benchmarking."""
    ASYNC = "async"
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    SEQUENTIAL = "sequential"


@dataclass
class ConcurrencyMetrics:
    """Metrics for a specific concurrency level."""
    concurrency_level: int
    execution_mode: ExecutionMode
    avg_execution_time: float
    std_execution_time: float
    min_execution_time: float
    max_execution_time: float
    throughput: float  # executions per second
    success_rate: float  # 0-1
    error_count: int
    resource_usage: Dict[str, float]  # CPU, memory, etc.


@dataclass
class ConcurrencyReport:
    """Comprehensive concurrency analysis report."""
    graph_name: str
    total_executions: int
    concurrency_metrics: List[ConcurrencyMetrics]
    optimal_concurrency: int
    performance_curve: Dict[int, float]  # concurrency -> throughput
    bottlenecks: List[str]
    recommendations: List[str]
    scalability_score: float  # 0-1


class ConcurrentBenchmark:
    """Advanced concurrent execution benchmarking for LangGraph.
    
    Tests various concurrency patterns and provides detailed analysis
    of performance characteristics under different load conditions.
    """
    
    def __init__(
        self,
        max_workers: int = None,
        warmup_runs: int = 3,
        test_runs: int = 5,
        timeout: float = 30.0
    ):
        self.max_workers = max_workers or mp.cpu_count()
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs
        self.timeout = timeout
        
    def benchmark_concurrency(
        self,
        graph: Pregel,
        input_data: dict,
        concurrency_levels: List[int] = None,
        execution_modes: List[ExecutionMode] = None,
        config: Optional[dict] = None
    ) -> ConcurrencyReport:
        """Comprehensive concurrency benchmarking.
        
        Args:
            graph: LangGraph to benchmark
            input_data: Input data for graph execution
            concurrency_levels: List of concurrency levels to test
            execution_modes: List of execution modes to test
            config: Optional configuration for graph execution
            
        Returns:
            Detailed concurrency analysis report
        """
        if concurrency_levels is None:
            concurrency_levels = [1, 2, 4, 8, 16, 32, 64]
            
        if execution_modes is None:
            execution_modes = [ExecutionMode.ASYNC, ExecutionMode.THREAD_POOL]
            
        all_metrics = []
        performance_curve = {}
        
        print(f"Starting concurrency benchmark with {len(concurrency_levels)} levels...")
        
        for concurrency in concurrency_levels:
            print(f"Testing concurrency level: {concurrency}")
            
            for mode in execution_modes:
                metrics = self._test_concurrency_level(
                    graph, input_data, concurrency, mode, config
                )
                all_metrics.append(metrics)
                
                # Update performance curve
                key = f"{mode.value}_{concurrency}"
                performance_curve[key] = metrics.throughput
                
        # Analyze results
        optimal_concurrency = self._find_optimal_concurrency(all_metrics)
        bottlenecks = self._identify_bottlenecks(all_metrics)
        recommendations = self._generate_recommendations(all_metrics, optimal_concurrency)
        scalability_score = self._calculate_scalability_score(all_metrics)
        
        return ConcurrencyReport(
            graph_name=getattr(graph, '__name__', 'unknown'),
            total_executions=sum(m.concurrency_level for m in all_metrics),
            concurrency_metrics=all_metrics,
            optimal_concurrency=optimal_concurrency,
            performance_curve=performance_curve,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            scalability_score=scalability_score
        )
        
    def _test_concurrency_level(
        self,
        graph: Pregel,
        input_data: dict,
        concurrency: int,
        mode: ExecutionMode,
        config: Optional[dict]
    ) -> ConcurrencyMetrics:
        """Test a specific concurrency level and execution mode."""
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            self._execute_single(graph, input_data, mode, config)
            
        # Test runs
        execution_times = []
        errors = 0
        
        for _ in range(self.test_runs):
            try:
                start_time = time.time()
                self._execute_concurrent(graph, input_data, concurrency, mode, config)
                end_time = time.time()
                execution_times.append(end_time - start_time)
            except Exception as e:
                errors += 1
                print(f"Error in concurrency test: {e}")
                
        # Calculate metrics
        if execution_times:
            avg_time = statistics.mean(execution_times)
            std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            min_time = min(execution_times)
            max_time = max(execution_times)
            throughput = concurrency / avg_time
            success_rate = (self.test_runs - errors) / self.test_runs
        else:
            avg_time = std_time = min_time = max_time = 0
            throughput = 0
            success_rate = 0
            
        # Resource usage (simplified)
        resource_usage = self._measure_resource_usage()
        
        return ConcurrencyMetrics(
            concurrency_level=concurrency,
            execution_mode=mode,
            avg_execution_time=avg_time,
            std_execution_time=std_time,
            min_execution_time=min_time,
            max_execution_time=max_time,
            throughput=throughput,
            success_rate=success_rate,
            error_count=errors,
            resource_usage=resource_usage
        )
        
    def _execute_single(
        self, 
        graph: Pregel, 
        input_data: dict, 
        mode: ExecutionMode, 
        config: Optional[dict]
    ) -> None:
        """Execute graph once in specified mode."""
        if mode == ExecutionMode.ASYNC:
            if hasattr(graph, 'astream'):
                asyncio.run(self._run_async(graph, input_data, config))
            else:
                # Fallback to sync
                list(graph.stream(input_data, config=config))
        else:
            # Sync execution
            list(graph.stream(input_data, config=config))
            
    def _execute_concurrent(
        self,
        graph: Pregel,
        input_data: dict,
        concurrency: int,
        mode: ExecutionMode,
        config: Optional[dict]
    ) -> None:
        """Execute graph with specified concurrency."""
        if mode == ExecutionMode.ASYNC:
            asyncio.run(self._run_concurrent_async(graph, input_data, concurrency, config))
        elif mode == ExecutionMode.THREAD_POOL:
            self._run_concurrent_threads(graph, input_data, concurrency, config)
        elif mode == ExecutionMode.PROCESS_POOL:
            self._run_concurrent_processes(graph, input_data, concurrency, config)
        else:  # SEQUENTIAL
            for _ in range(concurrency):
                self._execute_single(graph, input_data, mode, config)
                
    async def _run_async(self, graph: Pregel, input_data: dict, config: Optional[dict]):
        """Run graph asynchronously."""
        async for _ in graph.astream(input_data, config=config):
            pass
            
    async def _run_concurrent_async(
        self, 
        graph: Pregel, 
        input_data: dict, 
        concurrency: int, 
        config: Optional[dict]
    ):
        """Run multiple async executions concurrently."""
        tasks = [
            self._run_async(graph, input_data, config)
            for _ in range(concurrency)
        ]
        await asyncio.gather(*tasks)
        
    def _run_concurrent_threads(
        self, 
        graph: Pregel, 
        input_data: dict, 
        concurrency: int, 
        config: Optional[dict]
    ):
        """Run multiple executions in thread pool."""
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(self._execute_single, graph, input_data, ExecutionMode.SEQUENTIAL, config)
                for _ in range(concurrency)
            ]
            # Wait for all to complete
            for future in futures:
                future.result()
                
    def _run_concurrent_processes(
        self, 
        graph: Pregel, 
        input_data: dict, 
        concurrency: int, 
        config: Optional[dict]
    ):
        """Run multiple executions in process pool."""
        # Note: This is simplified - in practice, you'd need to handle
        # serialization of the graph and input data
        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(self._execute_single, graph, input_data, ExecutionMode.SEQUENTIAL, config)
                for _ in range(concurrency)
            ]
            # Wait for all to complete
            for future in futures:
                future.result()
                
    def _measure_resource_usage(self) -> Dict[str, float]:
        """Measure current resource usage."""
        try:
            import psutil
            process = psutil.Process()
            
            return {
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "threads": process.num_threads()
            }
        except ImportError:
            # Return mock data if psutil not available
            return {
                "cpu_percent": 50.0,
                "memory_mb": 100.0,
                "threads": 1
            }
        
    def _find_optimal_concurrency(self, metrics: List[ConcurrencyMetrics]) -> int:
        """Find the optimal concurrency level based on throughput."""
        if not metrics:
            return 1
            
        # Group by concurrency level and find best throughput
        by_level = defaultdict(list)
        for metric in metrics:
            by_level[metric.concurrency_level].append(metric.throughput)
            
        best_level = 1
        best_throughput = 0
        
        for level, throughputs in by_level.items():
            avg_throughput = statistics.mean(throughputs)
            if avg_throughput > best_throughput:
                best_throughput = avg_throughput
                best_level = level
                
        return best_level
        
    def _identify_bottlenecks(self, metrics: List[ConcurrencyMetrics]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Check for decreasing throughput at higher concurrency
        by_level = defaultdict(list)
        for metric in metrics:
            by_level[metric.concurrency_level].append(metric.throughput)
            
        levels = sorted(by_level.keys())
        for i in range(1, len(levels)):
            prev_throughput = statistics.mean(by_level[levels[i-1]])
            curr_throughput = statistics.mean(by_level[levels[i]])
            
            if curr_throughput < prev_throughput * 0.8:  # 20% decrease
                bottlenecks.append(f"Throughput degradation at concurrency {levels[i]}")
                
        # Check for high error rates
        high_error_metrics = [m for m in metrics if m.success_rate < 0.9]
        if high_error_metrics:
            bottlenecks.append("High error rate detected in concurrent execution")
            
        # Check for resource constraints
        high_cpu_metrics = [m for m in metrics if m.resource_usage.get("cpu_percent", 0) > 90]
        if high_cpu_metrics:
            bottlenecks.append("CPU utilization approaching limits")
            
        return bottlenecks
        
    def _generate_recommendations(
        self, 
        metrics: List[ConcurrencyMetrics], 
        optimal_concurrency: int
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if optimal_concurrency == 1:
            recommendations.append("Graph may not be suitable for concurrent execution")
        elif optimal_concurrency < 4:
            recommendations.append("Consider optimizing graph for better concurrency")
        else:
            recommendations.append(f"Optimal concurrency level is {optimal_concurrency}")
            
        # Check for async vs thread performance
        async_metrics = [m for m in metrics if m.execution_mode == ExecutionMode.ASYNC]
        thread_metrics = [m for m in metrics if m.execution_mode == ExecutionMode.THREAD_POOL]
        
        if async_metrics and thread_metrics:
            async_throughput = statistics.mean([m.throughput for m in async_metrics])
            thread_throughput = statistics.mean([m.throughput for m in thread_metrics])
            
            if async_throughput > thread_throughput * 1.2:
                recommendations.append("Async execution shows better performance than threading")
            elif thread_throughput > async_throughput * 1.2:
                recommendations.append("Threading shows better performance than async execution")
                
        return recommendations
        
    def _calculate_scalability_score(self, metrics: List[ConcurrencyMetrics]) -> float:
        """Calculate scalability score (0-1, higher is better)."""
        if not metrics:
            return 0.0
            
        # Group by concurrency level
        by_level = defaultdict(list)
        for metric in metrics:
            by_level[metric.concurrency_level].append(metric.throughput)
            
        levels = sorted(by_level.keys())
        if len(levels) < 2:
            return 0.5
            
        # Calculate scaling efficiency
        base_throughput = statistics.mean(by_level[levels[0]])
        max_throughput = max(statistics.mean(by_level[level]) for level in levels)
        
        if base_throughput == 0:
            return 0.0
            
        # Ideal scaling would be linear
        max_level = max(levels)
        ideal_throughput = base_throughput * max_level
        actual_throughput = max_throughput
        
        efficiency = min(1.0, actual_throughput / ideal_throughput)
        return efficiency
