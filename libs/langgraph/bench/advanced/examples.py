"""Examples demonstrating the advanced performance optimization suite.

This module provides comprehensive examples of how to use the performance
optimization tools for LangGraph applications.
"""

import asyncio
import time
from typing import Dict, List, Any

try:
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.pregel import Pregel
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError:
    # Mock classes for examples without full dependencies
    class MessagesState(dict):
        pass
    
    class HumanMessage:
        def __init__(self, content):
            self.content = content
    
    class AIMessage:
        def __init__(self, content):
            self.content = content
    
    class StateGraph:
        def __init__(self, state_schema):
            self.state_schema = state_schema
            self.nodes = {}
            self.edges = []
        
        def add_node(self, name, func):
            self.nodes[name] = func
        
        def add_edge(self, from_node, to_node):
            self.edges.append((from_node, to_node))
        
        def compile(self):
            return MockPregel(self)
    
    class MockPregel:
        def __init__(self, graph):
            self.graph = graph
        
        def stream(self, input_data, config=None):
            yield {"messages": [AIMessage(content="Mock response")]}
        
        def astream(self, input_data, config=None):
            async def async_gen():
                yield {"messages": [AIMessage(content="Mock response")]}
            return async_gen()
    
    START = "__start__"
    END = "__end__"

from .memory_profiler import GraphMemoryProfiler, MemoryProfiler
from .concurrent_benchmark import ConcurrentBenchmark, ExecutionMode
from .real_world_workloads import (
    RealWorldWorkloadSimulator,
    CustomerSupportWorkload,
    CodeAssistantWorkload
)
from .performance_analyzer import PerformanceAnalyzer
from .optimization_engine import GraphOptimizer, OptimizationStrategy


def create_example_graph():
    """Create an example graph for performance testing."""
    
    def simple_node(state: MessagesState) -> Dict[str, Any]:
        """Simple processing node."""
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            response = f"Processed: {last_message.content}"
            return {"messages": [AIMessage(content=response)]}
        return {}
    
    def complex_node(state: MessagesState) -> Dict[str, Any]:
        """Complex processing node with some computation."""
        messages = state.get("messages", [])
        if messages:
            # Simulate some computation
            content = messages[-1].content
            processed = content.upper()
            # Simulate processing time
            time.sleep(0.1)
            return {"messages": [AIMessage(content=f"Complex: {processed}")]}
        return {}
    
    def memory_intensive_node(state: MessagesState) -> Dict[str, Any]:
        """Memory intensive node."""
        messages = state.get("messages", [])
        if messages:
            # Create some memory usage
            large_data = [f"data_{i}" for i in range(1000)]
            content = messages[-1].content
            return {"messages": [AIMessage(content=f"Memory intensive: {content}")]}
        return {}
    
    # Build graph
    builder = StateGraph(MessagesState)
    builder.add_node("simple", simple_node)
    builder.add_node("complex", complex_node)
    builder.add_node("memory_intensive", memory_intensive_node)
    
    # Add edges
    builder.add_edge(START, "simple")
    builder.add_edge("simple", "complex")
    builder.add_edge("complex", "memory_intensive")
    builder.add_edge("memory_intensive", END)
    
    return builder.compile()


def example_memory_profiling():
    """Example of memory profiling."""
    print("=== Memory Profiling Example ===")
    
    # Create example graph
    graph = create_example_graph()
    
    # Create profiler
    profiler = GraphMemoryProfiler(
        MemoryProfiler(
            track_allocations=True,
            sample_rate=0.05  # Sample every 50ms
        )
    )
    
    # Test input
    input_data = {
        "messages": [HumanMessage(content="Hello, this is a test message for performance analysis")]
    }
    
    # Profile execution
    print("Profiling memory usage...")
    report = profiler.profile_graph_execution(graph, input_data)
    
    # Display results
    print(f"Peak Memory: {report.peak_memory:.2f} MB")
    print(f"Total Memory Used: {report.total_memory_used:.2f} MB")
    print(f"Memory Growth Rate: {report.memory_growth_rate:.2f} MB/sec")
    print(f"Allocation Rate: {report.allocation_rate:.2f} allocations/sec")
    print(f"GC Efficiency: {report.gc_efficiency:.2f}")
    print(f"Memory Leaks: {len(report.memory_leaks)}")
    
    if report.recommendations:
        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")


def example_concurrency_benchmarking():
    """Example of concurrency benchmarking."""
    print("\n=== Concurrency Benchmarking Example ===")
    
    # Create example graph
    graph = create_example_graph()
    
    # Create benchmark
    benchmark = ConcurrentBenchmark(
        max_workers=8,
        test_runs=3
    )
    
    # Test input
    input_data = {
        "messages": [HumanMessage(content="Concurrency test message")]
    }
    
    # Run benchmark
    print("Running concurrency benchmark...")
    report = benchmark.benchmark_concurrency(
        graph,
        input_data,
        concurrency_levels=[1, 2, 4, 8],
        execution_modes=[ExecutionMode.ASYNC, ExecutionMode.THREAD_POOL]
    )
    
    # Display results
    print(f"Graph: {report.graph_name}")
    print(f"Total Executions: {report.total_executions}")
    print(f"Optimal Concurrency: {report.optimal_concurrency}")
    print(f"Scalability Score: {report.scalability_score:.2f}")
    
    print("\nPerformance by Concurrency Level:")
    for metric in report.concurrency_metrics:
        print(f"  {metric.concurrency_level}: {metric.throughput:.2f} req/sec "
              f"({metric.execution_mode.value}, {metric.success_rate:.2f} success rate)")
    
    if report.bottlenecks:
        print("\nBottlenecks:")
        for bottleneck in report.bottlenecks:
            print(f"  - {bottleneck}")
    
    if report.recommendations:
        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")


def example_workload_simulation():
    """Example of real-world workload simulation."""
    print("\n=== Workload Simulation Example ===")
    
    # Create example graph
    graph = create_example_graph()
    
    # Create workload simulator
    simulator = RealWorldWorkloadSimulator()
    
    # Create customer support workload
    cs_workload = CustomerSupportWorkload()
    profile = cs_workload.create_profile(
        arrival_rate=2.0,  # 2 requests per second
        duration=10.0,     # 10 seconds
        burst_probability=0.2  # 20% chance of bursts
    )
    
    # Run simulation
    print("Running customer support workload simulation...")
    report = simulator.simulate_workload(graph, profile)
    
    # Display results
    print(f"Workload: {report.workload_name}")
    print(f"Total Requests: {report.total_requests}")
    print(f"Successful Requests: {report.successful_requests}")
    print(f"Success Rate: {report.successful_requests/report.total_requests:.2%}")
    print(f"Avg Execution Time: {report.avg_execution_time:.2f} seconds")
    print(f"P95 Execution Time: {report.p95_execution_time:.2f} seconds")
    print(f"Avg Throughput: {report.avg_throughput:.2f} req/sec")
    print(f"Peak Throughput: {report.peak_throughput:.2f} req/sec")
    print(f"Avg Memory Usage: {report.avg_memory_usage:.2f} MB")
    print(f"Peak Memory Usage: {report.peak_memory_usage:.2f} MB")
    
    if report.recommendations:
        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")


def example_performance_analysis():
    """Example of comprehensive performance analysis."""
    print("\n=== Performance Analysis Example ===")
    
    # Create example graph
    graph = create_example_graph()
    
    # Run individual analyses
    print("Running memory profiling...")
    memory_profiler = GraphMemoryProfiler()
    memory_report = memory_profiler.profile_graph_execution(
        graph, 
        {"messages": [HumanMessage(content="Analysis test")]}
    )
    
    print("Running concurrency benchmark...")
    concurrency_benchmark = ConcurrentBenchmark(test_runs=2)
    concurrency_report = concurrency_benchmark.benchmark_concurrency(
        graph,
        {"messages": [HumanMessage(content="Concurrency test")]},
        concurrency_levels=[1, 2, 4],
        execution_modes=[ExecutionMode.ASYNC]
    )
    
    print("Running workload simulation...")
    workload_simulator = RealWorldWorkloadSimulator()
    cs_workload = CustomerSupportWorkload()
    workload_profile = cs_workload.create_profile(arrival_rate=1.0, duration=5.0)
    workload_report = workload_simulator.simulate_workload(graph, workload_profile)
    
    # Run comprehensive analysis
    print("Running comprehensive performance analysis...")
    analyzer = PerformanceAnalyzer()
    performance_report = analyzer.analyze_performance(
        graph,
        memory_report=memory_report,
        concurrency_report=concurrency_report,
        workload_reports=[workload_report]
    )
    
    # Display results
    analyzer.print_summary(performance_report)


def example_graph_optimization():
    """Example of graph optimization."""
    print("\n=== Graph Optimization Example ===")
    
    # Create example graph
    graph = create_example_graph()
    
    # Create optimizer
    optimizer = GraphOptimizer()
    
    # Analyze optimization potential
    print("Analyzing optimization potential...")
    potential = optimizer.analyze_optimization_potential(graph)
    print("Optimization Potential:")
    for strategy, improvement in potential.items():
        print(f"  {strategy.value}: {improvement:.1%}")
    
    # Create optimization plan
    print("\nCreating optimization plan...")
    plan = optimizer.create_optimization_plan(graph, target_improvement=0.3)
    print(f"Selected strategies: {[s.value for s in plan.strategies]}")
    print(f"Expected improvement: {plan.total_expected_improvement:.1%}")
    print(f"Implementation order: {[s.value for s in plan.implementation_order]}")
    
    # Apply optimizations
    print("\nApplying optimizations...")
    optimized_graph, results = optimizer.optimize_graph(
        graph, 
        plan.strategies
    )
    
    # Display results
    print("Optimization Results:")
    total_improvement = 0
    for result in results:
        print(f"  {result.strategy.value}: {result.improvement_percentage:.1f}% improvement")
        total_improvement += result.improvement_percentage
    print(f"  Total Improvement: {total_improvement:.1f}%")
    
    # Generate report
    report = optimizer.generate_optimization_report(plan, results)
    print(f"\nOptimization Report:\n{report}")


def example_cli_usage():
    """Example of CLI usage."""
    print("\n=== CLI Usage Examples ===")
    
    print("Memory profiling:")
    print("  python -m langgraph.bench.advanced.cli_commands memory --graph my_graph.py --input '{\"messages\": [{\"role\": \"user\", \"content\": \"test\"}]}'")
    
    print("\nConcurrency benchmarking:")
    print("  python -m langgraph.bench.advanced.cli_commands concurrency --graph my_graph.py --concurrency-levels '1,2,4,8'")
    
    print("\nWorkload simulation:")
    print("  python -m langgraph.bench.advanced.cli_commands workload --graph my_graph.py --workload customer_support --duration 60")
    
    print("\nPerformance analysis:")
    print("  python -m langgraph.bench.advanced.cli_commands analyze --graph my_graph.py --memory-report memory.json --concurrency-report concurrency.json")
    
    print("\nGraph optimization:")
    print("  python -m langgraph.bench.advanced.cli_commands optimize --graph my_graph.py --strategies 'caching,parallel_execution'")


def run_all_examples():
    """Run all examples."""
    print("LangGraph Performance Optimization Suite - Examples")
    print("=" * 60)
    
    try:
        example_memory_profiling()
        example_concurrency_benchmarking()
        example_workload_simulation()
        example_performance_analysis()
        example_graph_optimization()
        example_cli_usage()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
