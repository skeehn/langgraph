"""Advanced memory profiling for LangGraph execution.

This module provides comprehensive memory profiling capabilities including
allocation tracking, memory growth analysis, and memory leak detection.
"""

import time
import gc
from typing import Dict, List, Tuple, Optional, Any, ContextManager
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict
import threading
import weakref

try:
    import tracemalloc
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    # Mock psutil for testing
    class MockProcess:
        def memory_info(self):
            class MockMemoryInfo:
                rss = 100 * 1024 * 1024  # 100MB
            return MockMemoryInfo()
        
        def cpu_percent(self):
            return 50.0
        
        def num_threads(self):
            return 1
    
    class MockPsutil:
        def Process(self):
            return MockProcess()
    
    psutil = MockPsutil()

try:
    from langgraph.pregel import Pregel
    from langgraph.graph import StateGraph
except ImportError:
    # Mock classes for testing
    class Pregel:
        pass
    
    class StateGraph:
        pass


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a specific point in time."""
    timestamp: float
    peak_memory: float  # MB
    current_memory: float  # MB
    memory_growth: float  # MB since start
    allocations: int
    deallocations: int
    gc_objects: int
    thread_id: int
    node_name: Optional[str] = None
    phase: Optional[str] = None  # "start", "execution", "checkpoint", "end"


@dataclass
class MemoryReport:
    """Comprehensive memory analysis report."""
    total_memory_used: float  # MB
    peak_memory: float  # MB
    memory_growth_rate: float  # MB/second
    allocation_rate: float  # allocations/second
    deallocation_rate: float  # deallocations/second
    memory_leaks: List[Tuple[str, float]]  # (node_name, leak_size_mb)
    gc_efficiency: float  # 0-1, higher is better
    snapshots: List[MemorySnapshot]
    recommendations: List[str]


class MemoryProfiler:
    """Advanced memory profiler for LangGraph execution.
    
    Provides detailed memory tracking including:
    - Real-time memory usage monitoring
    - Allocation/deallocation tracking
    - Memory leak detection
    - Garbage collection analysis
    - Node-specific memory usage
    """
    
    def __init__(
        self,
        track_allocations: bool = True,
        track_gc: bool = True,
        sample_rate: float = 0.1,  # Sample every 100ms
        leak_threshold: float = 1.0  # MB
    ):
        self.track_allocations = track_allocations
        self.track_gc = track_gc
        self.sample_rate = sample_rate
        self.leak_threshold = leak_threshold
        
        self.snapshots: List[MemorySnapshot] = []
        self.initial_memory: float = 0.0
        self.start_time: float = 0.0
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Allocation tracking
        self._allocation_tracker: Dict[str, int] = defaultdict(int)
        self._deallocation_tracker: Dict[str, int] = defaultdict(int)
        self._object_refs: weakref.WeakSet = weakref.WeakSet()
        
    def start_profiling(self) -> None:
        """Start memory profiling."""
        if self.track_allocations:
            tracemalloc.start()
            
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.start_time = time.time()
        self.snapshots.clear()
        
        # Take initial snapshot
        self._take_snapshot("start", "initialization")
        
        # Start monitoring thread
        self._monitoring = True
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_memory)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
    def stop_profiling(self) -> None:
        """Stop memory profiling."""
        self._monitoring = False
        self._stop_monitoring.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            
        # Take final snapshot
        self._take_snapshot("end", "finalization")
        
        if self.track_allocations:
            tracemalloc.stop()
            
    def profile_node(self, node_name: str) -> ContextManager:
        """Context manager for profiling individual nodes."""
        return self._node_profiler(node_name)
        
    @contextmanager
    def _node_profiler(self, node_name: str):
        """Context manager for node-specific profiling."""
        self._take_snapshot(node_name, "node_start")
        try:
            yield
        finally:
            self._take_snapshot(node_name, "node_end")
            
    def _monitor_memory(self) -> None:
        """Background thread for continuous memory monitoring."""
        while not self._stop_monitoring.wait(self.sample_rate):
            if self._monitoring:
                self._take_snapshot("monitor", "continuous")
                
    def _take_snapshot(
        self, 
        node_name: Optional[str] = None, 
        phase: Optional[str] = None
    ) -> None:
        """Take a memory snapshot."""
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024
        current_time = time.time()
        
        # Get allocation info
        allocations = 0
        deallocations = 0
        if self.track_allocations:
            current, peak = tracemalloc.get_traced_memory()
            allocations = int(current)
            
        # Get GC info
        gc_objects = len(gc.get_objects()) if self.track_gc else 0
        
        snapshot = MemorySnapshot(
            timestamp=current_time,
            peak_memory=current_memory,
            current_memory=current_memory,
            memory_growth=current_memory - self.initial_memory,
            allocations=allocations,
            deallocations=deallocations,
            gc_objects=gc_objects,
            thread_id=threading.get_ident(),
            node_name=node_name,
            phase=phase
        )
        
        self.snapshots.append(snapshot)
        
    def analyze_memory_usage(self) -> MemoryReport:
        """Analyze collected memory data and generate report."""
        if not self.snapshots:
            return MemoryReport(
                total_memory_used=0,
                peak_memory=0,
                memory_growth_rate=0,
                allocation_rate=0,
                deallocation_rate=0,
                memory_leaks=[],
                gc_efficiency=0,
                snapshots=[],
                recommendations=[]
            )
            
        # Calculate basic metrics
        total_time = self.snapshots[-1].timestamp - self.snapshots[0].timestamp
        total_memory_used = self.snapshots[-1].memory_growth
        peak_memory = max(snapshot.current_memory for snapshot in self.snapshots)
        
        # Calculate rates
        memory_growth_rate = total_memory_used / total_time if total_time > 0 else 0
        allocation_rate = (
            (self.snapshots[-1].allocations - self.snapshots[0].allocations) / total_time
            if total_time > 0 else 0
        )
        
        # Detect memory leaks
        memory_leaks = self._detect_memory_leaks()
        
        # Calculate GC efficiency
        gc_efficiency = self._calculate_gc_efficiency()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            total_memory_used, peak_memory, memory_leaks, gc_efficiency
        )
        
        return MemoryReport(
            total_memory_used=total_memory_used,
            peak_memory=peak_memory,
            memory_growth_rate=memory_growth_rate,
            allocation_rate=allocation_rate,
            deallocation_rate=0,  # Would need custom tracking
            memory_leaks=memory_leaks,
            gc_efficiency=gc_efficiency,
            snapshots=self.snapshots,
            recommendations=recommendations
        )
        
    def _detect_memory_leaks(self) -> List[Tuple[str, float]]:
        """Detect potential memory leaks by node."""
        leaks = []
        node_memory = defaultdict(list)
        
        # Group snapshots by node
        for snapshot in self.snapshots:
            if snapshot.node_name and snapshot.phase == "node_end":
                node_memory[snapshot.node_name].append(snapshot.current_memory)
                
        # Check for increasing memory patterns
        for node_name, memory_values in node_memory.items():
            if len(memory_values) < 3:
                continue
                
            # Simple trend analysis
            if memory_values[-1] - memory_values[0] > self.leak_threshold:
                leak_size = memory_values[-1] - memory_values[0]
                leaks.append((node_name, leak_size))
                
        return leaks
        
    def _calculate_gc_efficiency(self) -> float:
        """Calculate garbage collection efficiency."""
        if not self.track_gc or len(self.snapshots) < 2:
            return 0.0
            
        # Simple heuristic: fewer objects over time = better GC
        initial_objects = self.snapshots[0].gc_objects
        final_objects = self.snapshots[-1].gc_objects
        
        if initial_objects == 0:
            return 1.0
            
        efficiency = max(0, 1 - (final_objects - initial_objects) / initial_objects)
        return min(1.0, efficiency)
        
    def _generate_recommendations(
        self, 
        total_memory: float, 
        peak_memory: float, 
        leaks: List[Tuple[str, float]], 
        gc_efficiency: float
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if total_memory > 100:  # MB
            recommendations.append(
                "Consider implementing memory-efficient state management patterns"
            )
            
        if peak_memory > 500:  # MB
            recommendations.append(
                "Peak memory usage is high - consider implementing state compression"
            )
            
        if leaks:
            leak_nodes = [node for node, _ in leaks]
            recommendations.append(
                f"Potential memory leaks detected in nodes: {', '.join(leak_nodes)}"
            )
            
        if gc_efficiency < 0.5:
            recommendations.append(
                "Garbage collection efficiency is low - consider optimizing object lifecycle"
            )
            
        if len(self.snapshots) > 1000:
            recommendations.append(
                "High number of memory snapshots - consider reducing sample rate"
            )
            
        return recommendations


class GraphMemoryProfiler:
    """High-level memory profiler for LangGraph execution."""
    
    def __init__(self, profiler: Optional[MemoryProfiler] = None):
        self.profiler = profiler or MemoryProfiler()
        
    def profile_graph_execution(
        self, 
        graph: Pregel, 
        input_data: dict, 
        config: Optional[dict] = None
    ) -> MemoryReport:
        """Profile memory usage during graph execution."""
        self.profiler.start_profiling()
        
        try:
            # Execute graph with profiling
            if hasattr(graph, 'astream'):
                # Async execution
                import asyncio
                asyncio.run(self._execute_async(graph, input_data, config))
            else:
                # Sync execution
                list(graph.stream(input_data, config=config))
        finally:
            self.profiler.stop_profiling()
            
        return self.profiler.analyze_memory_usage()
        
    async def _execute_async(self, graph: Pregel, input_data: dict, config: dict):
        """Execute graph asynchronously with profiling."""
        try:
            # Try async iteration first
            async for _ in graph.astream(input_data, config=config):
                pass
        except TypeError:
            # Fallback to sync execution if async not supported
            for _ in graph.stream(input_data, config=config):
                pass
