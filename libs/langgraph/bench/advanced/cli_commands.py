"""CLI commands for performance optimization suite.

This module provides command-line interfaces for running performance
analysis, benchmarking, and optimization operations.
"""

import click
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

from .memory_profiler import MemoryProfiler, GraphMemoryProfiler
from .concurrent_benchmark import ConcurrentBenchmark, ExecutionMode
from .real_world_workloads import (
    RealWorldWorkloadSimulator, 
    WorkloadType,
    CustomerSupportWorkload,
    CodeAssistantWorkload,
    ResearchAgentWorkload,
    MultiAgentWorkload
)
from .performance_analyzer import PerformanceAnalyzer
from .optimization_engine import GraphOptimizer, OptimizationStrategy


@click.group()
def perf():
    """LangGraph Performance Optimization Suite."""
    pass


@perf.command()
@click.option("--graph", required=True, help="Path to graph module or graph object")
@click.option("--input", help="Input data file (JSON) or JSON string")
@click.option("--output", help="Output file for memory report")
@click.option("--track-allocations/--no-track-allocations", default=True, help="Track memory allocations")
@click.option("--sample-rate", default=0.1, help="Memory sampling rate in seconds")
def memory(graph: str, input: str, output: str, track_allocations: bool, sample_rate: float):
    """Profile memory usage of a LangGraph execution."""
    click.echo("Starting memory profiling...")
    
    # Load graph
    graph_obj = _load_graph(graph)
    if not graph_obj:
        click.echo("Error: Could not load graph", err=True)
        return
        
    # Load input data
    input_data = _load_input_data(input)
    if not input_data:
        click.echo("Error: Could not load input data", err=True)
        return
        
    # Create profiler
    profiler = GraphMemoryProfiler(
        MemoryProfiler(
            track_allocations=track_allocations,
            sample_rate=sample_rate
        )
    )
    
    # Run profiling
    try:
        report = profiler.profile_graph_execution(graph_obj, input_data)
        
        # Display results
        click.echo("\nMemory Profiling Results:")
        click.echo(f"Peak Memory: {report.peak_memory:.2f} MB")
        click.echo(f"Total Memory Used: {report.total_memory_used:.2f} MB")
        click.echo(f"Memory Growth Rate: {report.memory_growth_rate:.2f} MB/sec")
        click.echo(f"Allocation Rate: {report.allocation_rate:.2f} allocations/sec")
        click.echo(f"GC Efficiency: {report.gc_efficiency:.2f}")
        click.echo(f"Memory Leaks: {len(report.memory_leaks)}")
        
        if report.recommendations:
            click.echo("\nRecommendations:")
            for rec in report.recommendations:
                click.echo(f"  - {rec}")
                
        # Save report
        if output:
            with open(output, 'w') as f:
                json.dump({
                    "peak_memory": report.peak_memory,
                    "total_memory_used": report.total_memory_used,
                    "memory_growth_rate": report.memory_growth_rate,
                    "allocation_rate": report.allocation_rate,
                    "gc_efficiency": report.gc_efficiency,
                    "memory_leaks": report.memory_leaks,
                    "recommendations": report.recommendations
                }, f, indent=2)
            click.echo(f"\nReport saved to {output}")
            
    except Exception as e:
        click.echo(f"Error during profiling: {e}", err=True)


@perf.command()
@click.option("--graph", required=True, help="Path to graph module or graph object")
@click.option("--input", help="Input data file (JSON) or JSON string")
@click.option("--output", help="Output file for concurrency report")
@click.option("--concurrency-levels", default="1,2,4,8,16,32", help="Comma-separated concurrency levels")
@click.option("--execution-modes", default="async,thread_pool", help="Comma-separated execution modes")
@click.option("--test-runs", default=5, help="Number of test runs per concurrency level")
def concurrency(
    graph: str, 
    input: str, 
    output: str, 
    concurrency_levels: str, 
    execution_modes: str,
    test_runs: int
):
    """Benchmark concurrent execution performance."""
    click.echo("Starting concurrency benchmarking...")
    
    # Load graph
    graph_obj = _load_graph(graph)
    if not graph_obj:
        click.echo("Error: Could not load graph", err=True)
        return
        
    # Load input data
    input_data = _load_input_data(input)
    if not input_data:
        click.echo("Error: Could not load input data", err=True)
        return
        
    # Parse options
    concurrency_list = [int(x.strip()) for x in concurrency_levels.split(",")]
    execution_mode_list = [ExecutionMode(x.strip()) for x in execution_modes.split(",")]
    
    # Create benchmark
    benchmark = ConcurrentBenchmark(test_runs=test_runs)
    
    # Run benchmark
    try:
        report = benchmark.benchmark_concurrency(
            graph_obj, 
            input_data, 
            concurrency_levels=concurrency_list,
            execution_modes=execution_mode_list
        )
        
        # Display results
        click.echo(f"\nConcurrency Benchmark Results:")
        click.echo(f"Graph: {report.graph_name}")
        click.echo(f"Total Executions: {report.total_executions}")
        click.echo(f"Optimal Concurrency: {report.optimal_concurrency}")
        click.echo(f"Scalability Score: {report.scalability_score:.2f}")
        
        click.echo(f"\nPerformance by Concurrency Level:")
        for metric in report.concurrency_metrics:
            click.echo(f"  {metric.concurrency_level}: {metric.throughput:.2f} req/sec "
                      f"({metric.execution_mode.value}, {metric.success_rate:.2f} success rate)")
                      
        if report.bottlenecks:
            click.echo(f"\nBottlenecks:")
            for bottleneck in report.bottlenecks:
                click.echo(f"  - {bottleneck}")
                
        if report.recommendations:
            click.echo(f"\nRecommendations:")
            for rec in report.recommendations:
                click.echo(f"  - {rec}")
                
        # Save report
        if output:
            with open(output, 'w') as f:
                json.dump({
                    "graph_name": report.graph_name,
                    "total_executions": report.total_executions,
                    "optimal_concurrency": report.optimal_concurrency,
                    "scalability_score": report.scalability_score,
                    "bottlenecks": report.bottlenecks,
                    "recommendations": report.recommendations,
                    "metrics": [
                        {
                            "concurrency_level": m.concurrency_level,
                            "execution_mode": m.execution_mode.value,
                            "throughput": m.throughput,
                            "success_rate": m.success_rate,
                            "avg_execution_time": m.avg_execution_time
                        }
                        for m in report.concurrency_metrics
                    ]
                }, f, indent=2)
            click.echo(f"\nReport saved to {output}")
            
    except Exception as e:
        click.echo(f"Error during benchmarking: {e}", err=True)


@perf.command()
@click.option("--graph", required=True, help="Path to graph module or graph object")
@click.option("--workload", required=True, help="Workload type (customer_support, code_assistant, research_agent, multi_agent)")
@click.option("--output", help="Output file for workload report")
@click.option("--arrival-rate", default=1.0, help="Request arrival rate (requests per second)")
@click.option("--duration", default=60.0, help="Test duration in seconds")
@click.option("--burst-probability", default=0.1, help="Probability of request bursts")
def workload(
    graph: str, 
    workload: str, 
    output: str, 
    arrival_rate: float, 
    duration: float,
    burst_probability: float
):
    """Simulate real-world workload patterns."""
    click.echo(f"Starting {workload} workload simulation...")
    
    # Load graph
    graph_obj = _load_graph(graph)
    if not graph_obj:
        click.echo("Error: Could not load graph", err=True)
        return
        
    # Create workload simulator
    simulator = RealWorldWorkloadSimulator()
    
    # Get workload generator
    workload_generators = {
        "customer_support": CustomerSupportWorkload(),
        "code_assistant": CodeAssistantWorkload(),
        "research_agent": ResearchAgentWorkload(),
        "multi_agent": MultiAgentWorkload()
    }
    
    if workload not in workload_generators:
        click.echo(f"Error: Unknown workload type '{workload}'", err=True)
        click.echo(f"Available types: {', '.join(workload_generators.keys())}")
        return
        
    workload_generator = workload_generators[workload]
    
    # Create workload profile
    profile = workload_generator.create_profile(
        arrival_rate=arrival_rate,
        duration=duration,
        burst_probability=burst_probability
    )
    
    # Run simulation
    try:
        report = simulator.simulate_workload(graph_obj, profile)
        
        # Display results
        click.echo(f"\nWorkload Simulation Results:")
        click.echo(f"Workload: {report.workload_name}")
        click.echo(f"Total Requests: {report.total_requests}")
        click.echo(f"Successful Requests: {report.successful_requests}")
        click.echo(f"Failed Requests: {report.failed_requests}")
        click.echo(f"Success Rate: {report.successful_requests/report.total_requests:.2%}")
        click.echo(f"Avg Execution Time: {report.avg_execution_time:.2f} seconds")
        click.echo(f"P95 Execution Time: {report.p95_execution_time:.2f} seconds")
        click.echo(f"P99 Execution Time: {report.p99_execution_time:.2f} seconds")
        click.echo(f"Avg Throughput: {report.avg_throughput:.2f} req/sec")
        click.echo(f"Peak Throughput: {report.peak_throughput:.2f} req/sec")
        click.echo(f"Avg Memory Usage: {report.avg_memory_usage:.2f} MB")
        click.echo(f"Peak Memory Usage: {report.peak_memory_usage:.2f} MB")
        
        if report.recommendations:
            click.echo(f"\nRecommendations:")
            for rec in report.recommendations:
                click.echo(f"  - {rec}")
                
        # Save report
        if output:
            with open(output, 'w') as f:
                json.dump({
                    "workload_name": report.workload_name,
                    "total_requests": report.total_requests,
                    "successful_requests": report.successful_requests,
                    "failed_requests": report.failed_requests,
                    "success_rate": report.successful_requests/report.total_requests,
                    "avg_execution_time": report.avg_execution_time,
                    "p95_execution_time": report.p95_execution_time,
                    "p99_execution_time": report.p99_execution_time,
                    "avg_throughput": report.avg_throughput,
                    "peak_throughput": report.peak_throughput,
                    "avg_memory_usage": report.avg_memory_usage,
                    "peak_memory_usage": report.peak_memory_usage,
                    "recommendations": report.recommendations
                }, f, indent=2)
            click.echo(f"\nReport saved to {output}")
            
    except Exception as e:
        click.echo(f"Error during workload simulation: {e}", err=True)


@perf.command()
@click.option("--graph", required=True, help="Path to graph module or graph object")
@click.option("--memory-report", help="Path to memory report file")
@click.option("--concurrency-report", help="Path to concurrency report file")
@click.option("--workload-reports", help="Comma-separated paths to workload report files")
@click.option("--output", help="Output file for performance report")
def analyze(
    graph: str, 
    memory_report: str, 
    concurrency_report: str, 
    workload_reports: str,
    output: str
):
    """Perform comprehensive performance analysis."""
    click.echo("Starting comprehensive performance analysis...")
    
    # Load graph
    graph_obj = _load_graph(graph)
    if not graph_obj:
        click.echo("Error: Could not load graph", err=True)
        return
        
    # Load reports
    memory_data = _load_report(memory_report) if memory_report else None
    concurrency_data = _load_report(concurrency_report) if concurrency_report else None
    workload_data = [_load_report(path) for path in workload_reports.split(",")] if workload_reports else None
    
    # Create analyzer
    analyzer = PerformanceAnalyzer()
    
    # Run analysis
    try:
        # Convert loaded data to report objects (simplified)
        memory_report_obj = None  # Would convert from loaded data
        concurrency_report_obj = None  # Would convert from loaded data
        workload_report_objs = None  # Would convert from loaded data
        
        report = analyzer.analyze_performance(
            graph_obj,
            memory_report=memory_report_obj,
            concurrency_report=concurrency_report_obj,
            workload_reports=workload_report_objs
        )
        
        # Display summary
        analyzer.print_summary(report)
        
        # Save report
        if output:
            analyzer.export_report(report, output)
            click.echo(f"\nDetailed report saved to {output}")
            
    except Exception as e:
        click.echo(f"Error during analysis: {e}", err=True)


@perf.command()
@click.option("--graph", required=True, help="Path to graph module or graph object")
@click.option("--strategies", help="Comma-separated optimization strategies")
@click.option("--output", help="Output file for optimization report")
@click.option("--target-improvement", default=0.5, help="Target improvement percentage")
def optimize(
    graph: str, 
    strategies: str, 
    output: str, 
    target_improvement: float
):
    """Optimize graph performance using various strategies."""
    click.echo("Starting graph optimization...")
    
    # Load graph
    graph_obj = _load_graph(graph)
    if not graph_obj:
        click.echo("Error: Could not load graph", err=True)
        return
        
    # Create optimizer
    optimizer = GraphOptimizer()
    
    try:
        if strategies:
            # Use specified strategies
            strategy_list = [OptimizationStrategy(s.strip()) for s in strategies.split(",")]
        else:
            # Create optimization plan
            plan = optimizer.create_optimization_plan(graph_obj, target_improvement)
            strategy_list = plan.strategies
            click.echo(f"Optimization plan created with {len(strategy_list)} strategies")
            
        # Apply optimizations
        optimized_graph, results = optimizer.optimize_graph(graph_obj, strategy_list)
        
        # Display results
        click.echo(f"\nOptimization Results:")
        total_improvement = 0
        for result in results:
            click.echo(f"  {result.strategy.value}: {result.improvement_percentage:.1f}% improvement")
            total_improvement += result.improvement_percentage
            
        click.echo(f"  Total Improvement: {total_improvement:.1f}%")
        
        # Save report
        if output:
            report = optimizer.generate_optimization_report(
                optimizer.create_optimization_plan(graph_obj, target_improvement),
                results
            )
            with open(output, 'w') as f:
                f.write(report)
            click.echo(f"\nOptimization report saved to {output}")
            
    except Exception as e:
        click.echo(f"Error during optimization: {e}", err=True)


def _load_graph(graph_path: str):
    """Load graph from path or return None if not found."""
    # This is a simplified implementation
    # In practice, you'd implement proper graph loading
    try:
        # Try to import and get graph
        import importlib.util
        spec = importlib.util.spec_from_file_location("graph_module", graph_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, 'graph', None)
    except Exception:
        pass
    return None


def _load_input_data(input_data: str) -> Optional[dict]:
    """Load input data from file or JSON string."""
    if not input_data:
        return {"messages": [{"role": "user", "content": "Hello"}]}
        
    # Try to load from file
    if Path(input_data).exists():
        try:
            with open(input_data, 'r') as f:
                return json.load(f)
        except Exception:
            pass
            
    # Try to parse as JSON string
    try:
        return json.loads(input_data)
    except Exception:
        pass
        
    return None


def _load_report(report_path: str) -> Optional[dict]:
    """Load report from file."""
    try:
        with open(report_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


# PerformanceCLI class for programmatic access
class PerformanceCLI:
    """Programmatic interface to performance optimization suite."""
    
    def __init__(self):
        self.memory_profiler = GraphMemoryProfiler()
        self.concurrent_benchmark = ConcurrentBenchmark()
        self.workload_simulator = RealWorldWorkloadSimulator()
        self.performance_analyzer = PerformanceAnalyzer()
        self.optimizer = GraphOptimizer()
        
    def profile_memory(self, graph, input_data, **kwargs):
        """Profile memory usage."""
        return self.memory_profiler.profile_graph_execution(graph, input_data, **kwargs)
        
    def benchmark_concurrency(self, graph, input_data, **kwargs):
        """Benchmark concurrency."""
        return self.concurrent_benchmark.benchmark_concurrency(graph, input_data, **kwargs)
        
    def simulate_workload(self, graph, workload_profile, **kwargs):
        """Simulate workload."""
        return self.workload_simulator.simulate_workload(graph, workload_profile, **kwargs)
        
    def analyze_performance(self, graph, **kwargs):
        """Analyze performance."""
        return self.performance_analyzer.analyze_performance(graph, **kwargs)
        
    def optimize_graph(self, graph, strategies, **kwargs):
        """Optimize graph."""
        return self.optimizer.optimize_graph(graph, strategies, **kwargs)


if __name__ == "__main__":
    perf()
