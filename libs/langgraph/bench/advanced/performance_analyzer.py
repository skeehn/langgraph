"""Performance analysis and reporting for LangGraph applications.

This module provides comprehensive performance analysis capabilities including
bottleneck identification, optimization recommendations, and performance reporting.
"""

import time
import statistics
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from langgraph.pregel import Pregel
    from langgraph.graph import StateGraph
except ImportError:
    # Mock classes for testing
    class Pregel:
        pass
    
    class StateGraph:
        pass

from .memory_profiler import MemoryReport
from .concurrent_benchmark import ConcurrencyReport
from .real_world_workloads import WorkloadReport


class PerformanceMetric(Enum):
    """Types of performance metrics."""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    CONCURRENCY = "concurrency"


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    metric: PerformanceMetric
    warning_threshold: float
    critical_threshold: float
    unit: str


@dataclass
class Bottleneck:
    """Identified performance bottleneck."""
    name: str
    severity: str  # "low", "medium", "high", "critical"
    metric: PerformanceMetric
    current_value: float
    threshold_value: float
    impact: str
    recommendations: List[str]


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    category: str  # "memory", "cpu", "concurrency", "algorithm", "architecture"
    priority: int  # 1-10, higher is more important
    title: str
    description: str
    expected_improvement: str
    implementation_effort: str  # "low", "medium", "high"
    code_examples: List[str]


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report."""
    graph_name: str
    analysis_timestamp: float
    overall_score: float  # 0-100
    bottlenecks: List[Bottleneck]
    recommendations: List[OptimizationRecommendation]
    metrics_summary: Dict[str, Any]
    detailed_metrics: Dict[str, Any]
    comparison_data: Optional[Dict[str, Any]] = None
    visualizations: Optional[Dict[str, str]] = None  # Paths to generated charts


class PerformanceAnalyzer:
    """Advanced performance analyzer for LangGraph applications.
    
    Provides comprehensive analysis of performance characteristics,
    bottleneck identification, and optimization recommendations.
    """
    
    def __init__(self, thresholds: Optional[Dict[PerformanceMetric, PerformanceThreshold]] = None):
        self.thresholds = thresholds or self._get_default_thresholds()
        
    def _get_default_thresholds(self) -> Dict[PerformanceMetric, PerformanceThreshold]:
        """Get default performance thresholds."""
        return {
            PerformanceMetric.EXECUTION_TIME: PerformanceThreshold(
                metric=PerformanceMetric.EXECUTION_TIME,
                warning_threshold=5.0,  # seconds
                critical_threshold=10.0,
                unit="seconds"
            ),
            PerformanceMetric.MEMORY_USAGE: PerformanceThreshold(
                metric=PerformanceMetric.MEMORY_USAGE,
                warning_threshold=500.0,  # MB
                critical_threshold=1000.0,
                unit="MB"
            ),
            PerformanceMetric.CPU_USAGE: PerformanceThreshold(
                metric=PerformanceMetric.CPU_USAGE,
                warning_threshold=70.0,  # percentage
                critical_threshold=90.0,
                unit="percentage"
            ),
            PerformanceMetric.THROUGHPUT: PerformanceThreshold(
                metric=PerformanceMetric.THROUGHPUT,
                warning_threshold=1.0,  # requests per second
                critical_threshold=0.5,
                unit="requests/second"
            ),
            PerformanceMetric.ERROR_RATE: PerformanceThreshold(
                metric=PerformanceMetric.ERROR_RATE,
                warning_threshold=0.05,  # 5%
                critical_threshold=0.1,  # 10%
                unit="percentage"
            )
        }
        
    def analyze_performance(
        self,
        graph: Pregel,
        memory_report: Optional[MemoryReport] = None,
        concurrency_report: Optional[ConcurrencyReport] = None,
        workload_reports: Optional[List[WorkloadReport]] = None,
        baseline_data: Optional[Dict[str, Any]] = None
    ) -> PerformanceReport:
        """Perform comprehensive performance analysis.
        
        Args:
            graph: LangGraph to analyze
            memory_report: Memory profiling results
            concurrency_report: Concurrency benchmarking results
            workload_reports: Real-world workload test results
            baseline_data: Baseline performance data for comparison
            
        Returns:
            Comprehensive performance analysis report
        """
        print("Starting comprehensive performance analysis...")
        
        # Collect all metrics
        metrics_summary = self._collect_metrics_summary(
            memory_report, concurrency_report, workload_reports
        )
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(metrics_summary)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            bottlenecks, metrics_summary, baseline_data
        )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(metrics_summary, bottlenecks)
        
        # Generate detailed metrics
        detailed_metrics = self._generate_detailed_metrics(
            memory_report, concurrency_report, workload_reports
        )
        
        # Create visualizations
        visualizations = self._create_visualizations(
            memory_report, concurrency_report, workload_reports
        )
        
        return PerformanceReport(
            graph_name=getattr(graph, '__name__', 'unknown'),
            analysis_timestamp=time.time(),
            overall_score=overall_score,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            metrics_summary=metrics_summary,
            detailed_metrics=detailed_metrics,
            comparison_data=baseline_data,
            visualizations=visualizations
        )
        
    def _collect_metrics_summary(
        self,
        memory_report: Optional[MemoryReport],
        concurrency_report: Optional[ConcurrencyReport],
        workload_reports: Optional[List[WorkloadReport]]
    ) -> Dict[str, Any]:
        """Collect and summarize all performance metrics."""
        summary = {
            "memory": {},
            "concurrency": {},
            "workload": {},
            "overall": {}
        }
        
        # Memory metrics
        if memory_report:
            summary["memory"] = {
                "peak_memory_mb": memory_report.peak_memory,
                "total_memory_used_mb": memory_report.total_memory_used,
                "memory_growth_rate_mb_per_sec": memory_report.memory_growth_rate,
                "allocation_rate_per_sec": memory_report.allocation_rate,
                "gc_efficiency": memory_report.gc_efficiency,
                "memory_leaks": len(memory_report.memory_leaks)
            }
            
        # Concurrency metrics
        if concurrency_report:
            summary["concurrency"] = {
                "optimal_concurrency": concurrency_report.optimal_concurrency,
                "scalability_score": concurrency_report.scalability_score,
                "total_executions": concurrency_report.total_executions,
                "bottlenecks": len(concurrency_report.bottlenecks)
            }
            
        # Workload metrics
        if workload_reports:
            all_workloads = workload_reports
            summary["workload"] = {
                "total_requests": sum(wr.total_requests for wr in all_workloads),
                "success_rate": sum(wr.successful_requests for wr in all_workloads) / 
                               sum(wr.total_requests for wr in all_workloads) if all_workloads else 0,
                "avg_execution_time": statistics.mean([wr.avg_execution_time for wr in all_workloads]) if all_workloads else 0,
                "avg_throughput": statistics.mean([wr.avg_throughput for wr in all_workloads]) if all_workloads else 0,
                "peak_memory_usage": max([wr.peak_memory_usage for wr in all_workloads]) if all_workloads else 0
            }
            
        # Overall metrics
        summary["overall"] = {
            "analysis_timestamp": time.time(),
            "data_sources": len([r for r in [memory_report, concurrency_report] if r is not None]) + len(workload_reports or [])
        }
        
        return summary
        
    def _identify_bottlenecks(self, metrics_summary: Dict[str, Any]) -> List[Bottleneck]:
        """Identify performance bottlenecks based on metrics."""
        bottlenecks = []
        
        # Memory bottlenecks
        memory_metrics = metrics_summary.get("memory", {})
        if memory_metrics:
            peak_memory = memory_metrics.get("peak_memory_mb", 0)
            if peak_memory > self.thresholds[PerformanceMetric.MEMORY_USAGE].critical_threshold:
                bottlenecks.append(Bottleneck(
                    name="High Memory Usage",
                    severity="critical",
                    metric=PerformanceMetric.MEMORY_USAGE,
                    current_value=peak_memory,
                    threshold_value=self.thresholds[PerformanceMetric.MEMORY_USAGE].critical_threshold,
                    impact="May cause out-of-memory errors",
                    recommendations=[
                        "Implement memory-efficient state management",
                        "Consider state compression",
                        "Add memory monitoring and alerts"
                    ]
                ))
            elif peak_memory > self.thresholds[PerformanceMetric.MEMORY_USAGE].warning_threshold:
                bottlenecks.append(Bottleneck(
                    name="Moderate Memory Usage",
                    severity="medium",
                    metric=PerformanceMetric.MEMORY_USAGE,
                    current_value=peak_memory,
                    threshold_value=self.thresholds[PerformanceMetric.MEMORY_USAGE].warning_threshold,
                    impact="May impact performance under load",
                    recommendations=[
                        "Monitor memory usage patterns",
                        "Consider memory optimization"
                    ]
                ))
                
        # Concurrency bottlenecks
        concurrency_metrics = metrics_summary.get("concurrency", {})
        if concurrency_metrics:
            scalability_score = concurrency_metrics.get("scalability_score", 0)
            if scalability_score < 0.3:
                bottlenecks.append(Bottleneck(
                    name="Poor Scalability",
                    severity="high",
                    metric=PerformanceMetric.CONCURRENCY,
                    current_value=scalability_score,
                    threshold_value=0.5,
                    impact="Performance degrades significantly with concurrency",
                    recommendations=[
                        "Optimize graph for concurrent execution",
                        "Consider async patterns",
                        "Review state management for thread safety"
                    ]
                ))
                
        # Workload bottlenecks
        workload_metrics = metrics_summary.get("workload", {})
        if workload_metrics:
            success_rate = workload_metrics.get("success_rate", 1.0)
            if success_rate < 0.9:
                bottlenecks.append(Bottleneck(
                    name="High Error Rate",
                    severity="critical",
                    metric=PerformanceMetric.ERROR_RATE,
                    current_value=success_rate,
                    threshold_value=0.9,
                    impact="Many requests are failing",
                    recommendations=[
                        "Improve error handling",
                        "Add retry mechanisms",
                        "Review input validation"
                    ]
                ))
                
            avg_execution_time = workload_metrics.get("avg_execution_time", 0)
            if avg_execution_time > self.thresholds[PerformanceMetric.EXECUTION_TIME].critical_threshold:
                bottlenecks.append(Bottleneck(
                    name="Slow Execution",
                    severity="high",
                    metric=PerformanceMetric.EXECUTION_TIME,
                    current_value=avg_execution_time,
                    threshold_value=self.thresholds[PerformanceMetric.EXECUTION_TIME].critical_threshold,
                    impact="Poor user experience due to slow responses",
                    recommendations=[
                        "Profile and optimize slow nodes",
                        "Consider parallel execution",
                        "Implement caching strategies"
                    ]
                ))
                
        return bottlenecks
        
    def _generate_recommendations(
        self,
        bottlenecks: List[Bottleneck],
        metrics_summary: Dict[str, Any],
        baseline_data: Optional[Dict[str, Any]]
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Memory optimization recommendations
        memory_metrics = metrics_summary.get("memory", {})
        if memory_metrics.get("peak_memory_mb", 0) > 500:
            recommendations.append(OptimizationRecommendation(
                category="memory",
                priority=8,
                title="Implement Memory-Efficient State Management",
                description="Use streaming state updates and compression to reduce memory usage",
                expected_improvement="30-50% reduction in memory usage",
                implementation_effort="medium",
                code_examples=[
                    "# Use streaming state updates",
                    "def update_state_streaming(state, updates):",
                    "    for key, value in updates.items():",
                    "        if key in state:",
                    "            state[key] = compress_value(value)",
                    "        else:",
                    "            state[key] = value"
                ]
            ))
            
        # Concurrency optimization recommendations
        concurrency_metrics = metrics_summary.get("concurrency", {})
        if concurrency_metrics.get("scalability_score", 0) < 0.5:
            recommendations.append(OptimizationRecommendation(
                category="concurrency",
                priority=9,
                title="Optimize for Concurrent Execution",
                description="Implement async patterns and thread-safe state management",
                expected_improvement="2-5x improvement in concurrent throughput",
                implementation_effort="high",
                code_examples=[
                    "# Use async node functions",
                    "async def async_node(state):",
                    "    result = await process_async(state)",
                    "    return result",
                    "",
                    "# Add thread-safe state updates",
                    "import threading",
                    "state_lock = threading.Lock()",
                    "with state_lock:",
                    "    state.update(updates)"
                ]
            ))
            
        # Algorithm optimization recommendations
        workload_metrics = metrics_summary.get("workload", {})
        if workload_metrics.get("avg_execution_time", 0) > 5:
            recommendations.append(OptimizationRecommendation(
                category="algorithm",
                priority=7,
                title="Optimize Graph Execution Algorithm",
                description="Implement parallel node execution and smart caching",
                expected_improvement="40-60% reduction in execution time",
                implementation_effort="high",
                code_examples=[
                    "# Parallel node execution",
                    "async def execute_parallel_nodes(nodes, state):",
                    "    tasks = [node(state) for node in nodes]",
                    "    results = await asyncio.gather(*tasks)",
                    "    return merge_results(results)"
                ]
            ))
            
        # Architecture optimization recommendations
        if len(bottlenecks) > 3:
            recommendations.append(OptimizationRecommendation(
                category="architecture",
                priority=6,
                title="Consider Microservice Architecture",
                description="Break down complex graphs into smaller, focused services",
                expected_improvement="Better scalability and maintainability",
                implementation_effort="very_high",
                code_examples=[
                    "# Split into focused services",
                    "class CustomerService:",
                    "    def handle_support_request(self, request):",
                    "        # Handle customer support logic",
                    "        pass",
                    "",
                    "class CodeService:",
                    "    def handle_code_request(self, request):",
                    "        # Handle code assistance logic",
                    "        pass"
                ]
            ))
            
        return recommendations
        
    def _calculate_overall_score(
        self,
        metrics_summary: Dict[str, Any],
        bottlenecks: List[Bottleneck]
    ) -> float:
        """Calculate overall performance score (0-100)."""
        base_score = 100.0
        
        # Deduct points for bottlenecks
        for bottleneck in bottlenecks:
            if bottleneck.severity == "critical":
                base_score -= 25
            elif bottleneck.severity == "high":
                base_score -= 15
            elif bottleneck.severity == "medium":
                base_score -= 10
            elif bottleneck.severity == "low":
                base_score -= 5
                
        # Deduct points for poor metrics
        workload_metrics = metrics_summary.get("workload", {})
        if workload_metrics:
            success_rate = workload_metrics.get("success_rate", 1.0)
            if success_rate < 0.9:
                base_score -= 20
                
            avg_execution_time = workload_metrics.get("avg_execution_time", 0)
            if avg_execution_time > 10:
                base_score -= 15
                
        # Bonus points for good performance
        concurrency_metrics = metrics_summary.get("concurrency", {})
        if concurrency_metrics:
            scalability_score = concurrency_metrics.get("scalability_score", 0)
            if scalability_score > 0.8:
                base_score += 10
                
        return max(0.0, min(100.0, base_score))
        
    def _generate_detailed_metrics(
        self,
        memory_report: Optional[MemoryReport],
        concurrency_report: Optional[ConcurrencyReport],
        workload_reports: Optional[List[WorkloadReport]]
    ) -> Dict[str, Any]:
        """Generate detailed performance metrics."""
        detailed = {
            "memory_analysis": {},
            "concurrency_analysis": {},
            "workload_analysis": {}
        }
        
        if memory_report:
            detailed["memory_analysis"] = {
                "snapshots": [asdict(s) for s in memory_report.snapshots],
                "leak_analysis": memory_report.memory_leaks,
                "gc_analysis": {
                    "efficiency": memory_report.gc_efficiency,
                    "recommendations": memory_report.recommendations
                }
            }
            
        if concurrency_report:
            detailed["concurrency_analysis"] = {
                "metrics_by_level": [asdict(m) for m in concurrency_report.concurrency_metrics],
                "performance_curve": concurrency_report.performance_curve,
                "bottlenecks": concurrency_report.bottlenecks
            }
            
        if workload_reports:
            detailed["workload_analysis"] = {
                "workloads": [asdict(wr) for wr in workload_reports],
                "aggregate_metrics": self._calculate_aggregate_workload_metrics(workload_reports)
            }
            
        return detailed
        
    def _calculate_aggregate_workload_metrics(
        self, 
        workload_reports: List[WorkloadReport]
    ) -> Dict[str, Any]:
        """Calculate aggregate metrics across all workloads."""
        if not workload_reports:
            return {}
            
        return {
            "total_requests": sum(wr.total_requests for wr in workload_reports),
            "total_successful": sum(wr.successful_requests for wr in workload_reports),
            "overall_success_rate": sum(wr.successful_requests for wr in workload_reports) / 
                                   sum(wr.total_requests for wr in workload_reports),
            "avg_execution_time": statistics.mean([wr.avg_execution_time for wr in workload_reports]),
            "max_execution_time": max([wr.p99_execution_time for wr in workload_reports]),
            "avg_throughput": statistics.mean([wr.avg_throughput for wr in workload_reports]),
            "peak_throughput": max([wr.peak_throughput for wr in workload_reports]),
            "avg_memory_usage": statistics.mean([wr.avg_memory_usage for wr in workload_reports]),
            "peak_memory_usage": max([wr.peak_memory_usage for wr in workload_reports])
        }
        
    def _create_visualizations(
        self,
        memory_report: Optional[MemoryReport],
        concurrency_report: Optional[ConcurrencyReport],
        workload_reports: Optional[List[WorkloadReport]]
    ) -> Dict[str, str]:
        """Create performance visualization charts."""
        visualizations = {}
        
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, skipping visualizations")
            return visualizations
        
        try:
            # Memory usage over time
            if memory_report and memory_report.snapshots:
                self._create_memory_chart(memory_report)
                visualizations["memory_usage"] = "memory_usage.png"
                
            # Concurrency performance curve
            if concurrency_report:
                self._create_concurrency_chart(concurrency_report)
                visualizations["concurrency_curve"] = "concurrency_curve.png"
                
            # Workload performance comparison
            if workload_reports:
                self._create_workload_chart(workload_reports)
                visualizations["workload_comparison"] = "workload_comparison.png"
                
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
            
        return visualizations
        
    def _create_memory_chart(self, memory_report: MemoryReport):
        """Create memory usage over time chart."""
        snapshots = memory_report.snapshots
        timestamps = [s.timestamp for s in snapshots]
        memory_usage = [s.current_memory for s in snapshots]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, memory_usage, 'b-', linewidth=2)
        plt.title('Memory Usage Over Time')
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (MB)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('memory_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_concurrency_chart(self, concurrency_report: ConcurrencyReport):
        """Create concurrency performance curve chart."""
        metrics = concurrency_report.concurrency_metrics
        concurrency_levels = [m.concurrency_level for m in metrics]
        throughputs = [m.throughput for m in metrics]
        
        plt.figure(figsize=(10, 6))
        plt.plot(concurrency_levels, throughputs, 'ro-', linewidth=2, markersize=8)
        plt.title('Throughput vs Concurrency Level')
        plt.xlabel('Concurrency Level')
        plt.ylabel('Throughput (requests/second)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('concurrency_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_workload_chart(self, workload_reports: List[WorkloadReport]):
        """Create workload performance comparison chart."""
        workload_names = [wr.workload_name for wr in workload_reports]
        execution_times = [wr.avg_execution_time for wr in workload_reports]
        throughputs = [wr.avg_throughput for wr in workload_reports]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Execution time comparison
        ax1.bar(workload_names, execution_times, color='skyblue', alpha=0.7)
        ax1.set_title('Average Execution Time by Workload')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        ax2.bar(workload_names, throughputs, color='lightcoral', alpha=0.7)
        ax2.set_title('Average Throughput by Workload')
        ax2.set_ylabel('Throughput (requests/second)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('workload_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def export_report(self, report: PerformanceReport, filepath: str) -> None:
        """Export performance report to file."""
        report_dict = asdict(report)
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
            
        print(f"Performance report exported to {filepath}")
        
    def print_summary(self, report: PerformanceReport) -> None:
        """Print a summary of the performance report."""
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("="*60)
        print(f"Graph: {report.graph_name}")
        print(f"Overall Score: {report.overall_score:.1f}/100")
        print(f"Bottlenecks Found: {len(report.bottlenecks)}")
        print(f"Recommendations: {len(report.recommendations)}")
        
        if report.bottlenecks:
            print("\nBOTTLENECKS:")
            for bottleneck in report.bottlenecks:
                print(f"  - {bottleneck.name} ({bottleneck.severity})")
                print(f"    Impact: {bottleneck.impact}")
                
        if report.recommendations:
            print("\nTOP RECOMMENDATIONS:")
            sorted_recs = sorted(report.recommendations, key=lambda x: x.priority, reverse=True)
            for rec in sorted_recs[:3]:
                print(f"  - {rec.title} (Priority: {rec.priority})")
                print(f"    {rec.description}")
                
        print("="*60)
