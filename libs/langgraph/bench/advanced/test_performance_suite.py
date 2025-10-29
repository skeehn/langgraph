"""Basic tests for the performance optimization suite."""

import sys
import os
import time
from typing import Dict, Any

# Add the langgraph lib to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError:
    # Mock classes for testing without full dependencies
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
            # Simple mock implementation
            yield {"messages": [AIMessage(content="Mock response")]}
        
        def astream(self, input_data, config=None):
            # Simple mock implementation that returns an async generator
            async def async_gen():
                yield {"messages": [AIMessage(content="Mock response")]}
            return async_gen()
    
    # Define constants
    START = "__start__"
    END = "__end__"

from .memory_profiler import MemoryProfiler, GraphMemoryProfiler
from .concurrent_benchmark import ConcurrentBenchmark, ExecutionMode
from .real_world_workloads import CustomerSupportWorkload, RealWorldWorkloadSimulator
from .performance_analyzer import PerformanceAnalyzer
from .optimization_engine import GraphOptimizer, OptimizationStrategy


def create_test_graph() -> StateGraph:
    """Create a simple test graph."""
    
    def test_node(state: MessagesState) -> Dict[str, Any]:
        """Simple test node."""
        messages = state.get("messages", [])
        if messages:
            content = messages[-1].content
            return {"messages": [AIMessage(content=f"Processed: {content}")]}
        return {}
    
    builder = StateGraph(MessagesState)
    builder.add_node("test_node", test_node)
    builder.add_edge(START, "test_node")
    builder.add_edge("test_node", END)
    
    return builder


def test_memory_profiler():
    """Test memory profiler functionality."""
    graph = create_test_graph().compile()
    profiler = GraphMemoryProfiler(MemoryProfiler(track_allocations=False))
    
    input_data = {"messages": [HumanMessage(content="Test message")]}
    report = profiler.profile_graph_execution(graph, input_data)
    
    assert report.peak_memory > 0
    assert report.total_memory_used >= 0
    assert isinstance(report.recommendations, list)


def test_concurrent_benchmark():
    """Test concurrent benchmark functionality."""
    graph = create_test_graph().compile()
    benchmark = ConcurrentBenchmark(test_runs=2)
    
    input_data = {"messages": [HumanMessage(content="Test message")]}
    report = benchmark.benchmark_concurrency(
        graph,
        input_data,
        concurrency_levels=[1, 2],
        execution_modes=[ExecutionMode.ASYNC]
    )
    
    assert report.total_executions > 0
    assert report.optimal_concurrency >= 1
    assert len(report.concurrency_metrics) > 0


def test_workload_simulation():
    """Test workload simulation functionality."""
    graph = create_test_graph().compile()
    simulator = RealWorldWorkloadSimulator()
    workload = CustomerSupportWorkload()
    
    profile = workload.create_profile(
        arrival_rate=1.0,
        duration=2.0,  # Short duration for testing
        burst_probability=0.0
    )
    
    report = simulator.simulate_workload(graph, profile)
    
    assert report.total_requests >= 0
    assert report.successful_requests >= 0
    assert isinstance(report.recommendations, list)


def test_performance_analyzer():
    """Test performance analyzer functionality."""
    graph = create_test_graph().compile()
    analyzer = PerformanceAnalyzer()
    
    # Create mock reports
    memory_report = None  # Would be created by memory profiler
    concurrency_report = None  # Would be created by concurrent benchmark
    workload_reports = None  # Would be created by workload simulator
    
    report = analyzer.analyze_performance(
        graph,
        memory_report=memory_report,
        concurrency_report=concurrency_report,
        workload_reports=workload_reports
    )
    
    assert report.overall_score >= 0
    assert report.overall_score <= 100
    assert isinstance(report.bottlenecks, list)
    assert isinstance(report.recommendations, list)


def test_graph_optimizer():
    """Test graph optimizer functionality."""
    graph = create_test_graph().compile()
    optimizer = GraphOptimizer()
    
    # Test optimization potential analysis
    potential = optimizer.analyze_optimization_potential(graph)
    assert isinstance(potential, dict)
    assert len(potential) > 0
    
    # Test optimization plan creation
    plan = optimizer.create_optimization_plan(graph, target_improvement=0.1)
    assert isinstance(plan.strategies, list)
    assert isinstance(plan.implementation_order, list)
    assert plan.total_expected_improvement >= 0


def test_optimization_strategies():
    """Test individual optimization strategies."""
    graph = create_test_graph().compile()
    optimizer = GraphOptimizer()
    
    # Test parallel execution optimizer
    parallel_optimizer = optimizer.optimizers[OptimizationStrategy.PARALLEL_EXECUTION]
    assert parallel_optimizer.can_optimize(graph)
    assert parallel_optimizer.estimate_improvement(graph) >= 0
    
    # Test caching optimizer
    caching_optimizer = optimizer.optimizers[OptimizationStrategy.CACHING]
    assert caching_optimizer.can_optimize(graph)
    assert caching_optimizer.estimate_improvement(graph) >= 0


def test_workload_generators():
    """Test workload generator functionality."""
    # Test customer support workload
    cs_workload = CustomerSupportWorkload()
    request = cs_workload.generate_request()
    
    assert request.request_id.startswith("cs_")
    assert "messages" in request.input_data
    assert isinstance(request.expected_tools, list)
    assert request.complexity in ["low", "medium", "high"]
    
    # Test profile creation
    profile = cs_workload.create_profile(
        arrival_rate=2.0,
        duration=10.0,
        burst_probability=0.1
    )
    
    assert profile.name == "Customer Support"
    assert profile.arrival_rate == 2.0
    assert profile.duration == 10.0


def test_performance_metrics():
    """Test performance metric calculations."""
    from .performance_analyzer import PerformanceMetric, PerformanceThreshold
    
    # Test threshold creation
    threshold = PerformanceThreshold(
        metric=PerformanceMetric.EXECUTION_TIME,
        warning_threshold=5.0,
        critical_threshold=10.0,
        unit="seconds"
    )
    
    assert threshold.metric == PerformanceMetric.EXECUTION_TIME
    assert threshold.warning_threshold == 5.0
    assert threshold.critical_threshold == 10.0


def test_memory_snapshot():
    """Test memory snapshot functionality."""
    from .memory_profiler import MemorySnapshot
    
    snapshot = MemorySnapshot(
        timestamp=time.time(),
        peak_memory=100.0,
        current_memory=80.0,
        memory_growth=20.0,
        allocations=1000,
        deallocations=950,
        gc_objects=5000,
        thread_id=12345,
        node_name="test_node",
        phase="execution"
    )
    
    assert snapshot.peak_memory == 100.0
    assert snapshot.current_memory == 80.0
    assert snapshot.memory_growth == 20.0
    assert snapshot.node_name == "test_node"


def test_concurrency_metrics():
    """Test concurrency metrics functionality."""
    from .concurrent_benchmark import ConcurrencyMetrics, ExecutionMode
    
    metrics = ConcurrencyMetrics(
        concurrency_level=4,
        execution_mode=ExecutionMode.ASYNC,
        avg_execution_time=1.5,
        std_execution_time=0.2,
        min_execution_time=1.0,
        max_execution_time=2.0,
        throughput=2.67,
        success_rate=0.95,
        error_count=1,
        resource_usage={"cpu_percent": 50.0, "memory_mb": 200.0, "threads": 4}
    )
    
    assert metrics.concurrency_level == 4
    assert metrics.execution_mode == ExecutionMode.ASYNC
    assert metrics.throughput == 2.67
    assert metrics.success_rate == 0.95


if __name__ == "__main__":
    # Run basic tests
    print("Running performance suite tests...")
    
    try:
        test_memory_profiler()
        print("✓ Memory profiler test passed")
        
        test_concurrent_benchmark()
        print("✓ Concurrent benchmark test passed")
        
        test_workload_simulation()
        print("✓ Workload simulation test passed")
        
        test_performance_analyzer()
        print("✓ Performance analyzer test passed")
        
        test_graph_optimizer()
        print("✓ Graph optimizer test passed")
        
        test_optimization_strategies()
        print("✓ Optimization strategies test passed")
        
        test_workload_generators()
        print("✓ Workload generators test passed")
        
        test_performance_metrics()
        print("✓ Performance metrics test passed")
        
        test_memory_snapshot()
        print("✓ Memory snapshot test passed")
        
        test_concurrency_metrics()
        print("✓ Concurrency metrics test passed")
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
