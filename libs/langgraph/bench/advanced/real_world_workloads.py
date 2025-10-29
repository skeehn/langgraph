"""Real-world workload simulation for LangGraph performance testing.

This module provides realistic workload patterns that mirror actual
production usage of LangGraph applications.
"""

import random
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import json

try:
    from langgraph.pregel import Pregel
except ImportError:
    # Mock class for testing
    class Pregel:
        pass


class WorkloadType(Enum):
    """Types of real-world workloads."""
    CUSTOMER_SUPPORT = "customer_support"
    CODE_ASSISTANT = "code_assistant"
    RESEARCH_AGENT = "research_agent"
    MULTI_AGENT_COLLABORATION = "multi_agent_collaboration"
    CHATBOT = "chatbot"
    DATA_PROCESSING = "data_processing"


@dataclass
class WorkloadRequest:
    """Individual request in a workload."""
    request_id: str
    input_data: dict
    expected_tools: List[str]
    complexity: str  # "low", "medium", "high"
    priority: int  # 1-10, higher is more urgent
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class WorkloadProfile:
    """Profile defining a workload pattern."""
    name: str
    workload_type: WorkloadType
    request_pattern: Callable[[], WorkloadRequest]
    arrival_rate: float  # requests per second
    duration: float  # seconds
    burst_probability: float  # 0-1, probability of burst
    burst_multiplier: float  # multiplier for burst rate


@dataclass
class WorkloadResult:
    """Result of workload execution."""
    request_id: str
    execution_time: float
    success: bool
    error_message: Optional[str]
    memory_usage: float
    cpu_usage: float
    tools_used: List[str]
    response_quality: float  # 0-1, subjective quality


@dataclass
class WorkloadReport:
    """Comprehensive workload analysis report."""
    workload_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_execution_time: float
    p95_execution_time: float
    p99_execution_time: float
    avg_throughput: float
    peak_throughput: float
    avg_memory_usage: float
    peak_memory_usage: float
    avg_cpu_usage: float
    peak_cpu_usage: float
    results: List[WorkloadResult]
    recommendations: List[str]


class BaseWorkload(ABC):
    """Base class for workload generators."""
    
    def __init__(self, name: str, workload_type: WorkloadType):
        self.name = name
        self.workload_type = workload_type
        
    @abstractmethod
    def generate_request(self) -> WorkloadRequest:
        """Generate a single workload request."""
        pass
        
    def create_profile(
        self, 
        arrival_rate: float, 
        duration: float,
        burst_probability: float = 0.1,
        burst_multiplier: float = 3.0
    ) -> WorkloadProfile:
        """Create a workload profile."""
        return WorkloadProfile(
            name=self.name,
            workload_type=self.workload_type,
            request_pattern=self.generate_request,
            arrival_rate=arrival_rate,
            duration=duration,
            burst_probability=burst_probability,
            burst_multiplier=burst_multiplier
        )


class CustomerSupportWorkload(BaseWorkload):
    """Customer support agent workload simulation."""
    
    def __init__(self):
        super().__init__("Customer Support", WorkloadType.CUSTOMER_SUPPORT)
        
        # Common customer issues
        self.issue_templates = [
            "I'm having trouble with {product} - {issue}",
            "Can you help me with {product}? {issue}",
            "I need assistance with {product} because {issue}",
            "There's a problem with my {product}: {issue}",
            "I can't figure out how to {action} in {product}"
        ]
        
        self.products = [
            "account", "billing", "shipping", "returns", "login", 
            "password", "subscription", "payment", "order", "refund"
        ]
        
        self.issues = [
            "it's not working", "I can't access it", "it's slow", 
            "I'm getting an error", "it's confusing", "I need a refund",
            "it's not what I expected", "I can't find my order",
            "the price is wrong", "I want to cancel"
        ]
        
        self.actions = [
            "reset my password", "change my email", "update my address",
            "cancel my order", "track my package", "get a refund",
            "upgrade my plan", "downgrade my plan", "add a payment method"
        ]
        
    def generate_request(self) -> WorkloadRequest:
        """Generate a customer support request."""
        template = random.choice(self.issue_templates)
        product = random.choice(self.products)
        
        if "{issue}" in template:
            issue = random.choice(self.issues)
            content = template.format(product=product, issue=issue)
        else:
            action = random.choice(self.actions)
            content = template.format(product=product, action=action)
            
        # Add some context
        context = random.choice([
            "I've been a customer for 2 years",
            "This is my first time using this",
            "I'm a premium subscriber",
            "I'm on the free plan",
            "I've tried everything"
        ])
        
        full_content = f"{content}. {context}."
        
        return WorkloadRequest(
            request_id=f"cs_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            input_data={
                "messages": [{"role": "user", "content": full_content}]
            },
            expected_tools=["search_knowledge_base", "escalate_to_human", "create_ticket"],
            complexity=random.choices(["low", "medium", "high"], weights=[0.6, 0.3, 0.1])[0],
            priority=random.randint(1, 10),
            timestamp=time.time(),
            metadata={
                "product": product,
                "urgency": random.choice(["low", "medium", "high"]),
                "customer_tier": random.choice(["free", "premium", "enterprise"])
            }
        )


class CodeAssistantWorkload(BaseWorkload):
    """Code assistant workload simulation."""
    
    def __init__(self):
        super().__init__("Code Assistant", WorkloadType.CODE_ASSISTANT)
        
        self.languages = ["python", "javascript", "typescript", "java", "go", "rust", "c++"]
        self.task_types = [
            "debug this code", "optimize this function", "add error handling",
            "refactor this class", "write unit tests", "explain this algorithm",
            "convert to async", "add type hints", "implement this feature",
            "fix this bug", "improve performance", "add documentation"
        ]
        
        self.code_snippets = [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "class User:\n    def __init__(self, name, email):\n        self.name = name\n        self.email = email",
            "async def fetch_data(url):\n    response = await requests.get(url)\n    return response.json()",
            "def process_items(items):\n    result = []\n    for item in items:\n        if item > 0:\n            result.append(item * 2)\n    return result"
        ]
        
    def generate_request(self) -> WorkloadRequest:
        """Generate a code assistant request."""
        language = random.choice(self.languages)
        task_type = random.choice(self.task_types)
        code_snippet = random.choice(self.code_snippets)
        
        content = f"I need help with {language} code. {task_type}:\n\n```{language}\n{code_snippet}\n```"
        
        # Add additional context
        context_options = [
            "This is for a production system",
            "I'm learning this language",
            "This is for a school project",
            "I need this for work",
            "This is part of a larger refactoring"
        ]
        
        full_content = f"{content}\n\n{random.choice(context_options)}."
        
        return WorkloadRequest(
            request_id=f"ca_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            input_data={
                "messages": [{"role": "user", "content": full_content}]
            },
            expected_tools=["code_analyzer", "test_generator", "documentation_generator"],
            complexity=random.choices(["low", "medium", "high"], weights=[0.3, 0.5, 0.2])[0],
            priority=random.randint(1, 10),
            timestamp=time.time(),
            metadata={
                "language": language,
                "task_type": task_type,
                "code_length": len(code_snippet),
                "experience_level": random.choice(["beginner", "intermediate", "expert"])
            }
        )


class ResearchAgentWorkload(BaseWorkload):
    """Research agent workload simulation."""
    
    def __init__(self):
        super().__init__("Research Agent", WorkloadType.RESEARCH_AGENT)
        
        self.research_topics = [
            "artificial intelligence", "machine learning", "quantum computing",
            "blockchain technology", "renewable energy", "space exploration",
            "biotechnology", "nanotechnology", "robotics", "cybersecurity"
        ]
        
        self.question_types = [
            "What are the latest developments in {topic}?",
            "How does {topic} work?",
            "What are the applications of {topic}?",
            "What are the challenges in {topic}?",
            "What is the future of {topic}?",
            "Compare {topic} with {topic2}",
            "What are the ethical implications of {topic}?",
            "How is {topic} being used in {industry}?"
        ]
        
        self.industries = [
            "healthcare", "finance", "education", "manufacturing", 
            "transportation", "entertainment", "agriculture", "retail"
        ]
        
    def generate_request(self) -> WorkloadRequest:
        """Generate a research request."""
        topic = random.choice(self.research_topics)
        question_type = random.choice(self.question_types)
        
        if "{topic2}" in question_type:
            topic2 = random.choice(self.research_topics)
            while topic2 == topic:
                topic2 = random.choice(self.research_topics)
            content = question_type.format(topic=topic, topic2=topic2)
        elif "{industry}" in question_type:
            industry = random.choice(self.industries)
            content = question_type.format(topic=topic, industry=industry)
        else:
            content = question_type.format(topic=topic)
            
        # Add depth requirements
        depth = random.choice([
            "Give me a high-level overview",
            "I need detailed technical information",
            "Provide a comprehensive analysis",
            "Focus on recent developments",
            "Include historical context"
        ])
        
        full_content = f"{content} {depth}."
        
        return WorkloadRequest(
            request_id=f"ra_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            input_data={
                "messages": [{"role": "user", "content": full_content}]
            },
            expected_tools=["web_search", "academic_search", "fact_checker"],
            complexity=random.choices(["low", "medium", "high"], weights=[0.2, 0.4, 0.4])[0],
            priority=random.randint(1, 10),
            timestamp=time.time(),
            metadata={
                "topic": topic,
                "question_type": question_type,
                "depth_requirement": depth,
                "urgency": random.choice(["low", "medium", "high"])
            }
        )


class MultiAgentWorkload(BaseWorkload):
    """Multi-agent collaboration workload simulation."""
    
    def __init__(self):
        super().__init__("Multi-Agent Collaboration", WorkloadType.MULTI_AGENT_COLLABORATION)
        
        self.collaboration_scenarios = [
            "Design a new product feature",
            "Analyze market trends and create a strategy",
            "Solve a complex technical problem",
            "Plan and execute a project",
            "Review and improve existing processes",
            "Create comprehensive documentation",
            "Develop a training program",
            "Conduct a risk assessment"
        ]
        
        self.agent_roles = [
            "project_manager", "technical_lead", "designer", "analyst",
            "researcher", "writer", "reviewer", "coordinator"
        ]
        
    def generate_request(self) -> WorkloadRequest:
        """Generate a multi-agent collaboration request."""
        scenario = random.choice(self.collaboration_scenarios)
        num_agents = random.randint(3, 8)
        roles = random.sample(self.agent_roles, min(num_agents, len(self.agent_roles)))
        
        content = f"Collaborate on: {scenario}. Involve {num_agents} agents with roles: {', '.join(roles)}."
        
        # Add specific requirements
        requirements = random.choice([
            "Each agent should provide their perspective",
            "Agents should build on each other's work",
            "Include a final synthesis step",
            "Ensure all viewpoints are considered",
            "Create a detailed action plan"
        ])
        
        full_content = f"{content} {requirements}."
        
        return WorkloadRequest(
            request_id=f"ma_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            input_data={
                "messages": [{"role": "user", "content": full_content}],
                "agent_roles": roles,
                "collaboration_mode": "hierarchical"
            },
            expected_tools=["agent_coordinator", "consensus_builder", "synthesis_engine"],
            complexity=random.choices(["low", "medium", "high"], weights=[0.1, 0.3, 0.6])[0],
            priority=random.randint(1, 10),
            timestamp=time.time(),
            metadata={
                "scenario": scenario,
                "num_agents": num_agents,
                "roles": roles,
                "collaboration_type": random.choice(["hierarchical", "peer-to-peer", "hybrid"])
            }
        )


class RealWorldWorkloadSimulator:
    """Simulator for real-world workload patterns."""
    
    def __init__(self):
        self.workloads = {
            WorkloadType.CUSTOMER_SUPPORT: CustomerSupportWorkload(),
            WorkloadType.CODE_ASSISTANT: CodeAssistantWorkload(),
            WorkloadType.RESEARCH_AGENT: ResearchAgentWorkload(),
            WorkloadType.MULTI_AGENT_COLLABORATION: MultiAgentWorkload()
        }
        
    def simulate_workload(
        self,
        graph: Pregel,
        workload_profile: WorkloadProfile,
        config: Optional[dict] = None
    ) -> WorkloadReport:
        """Simulate a workload and measure performance."""
        print(f"Simulating {workload_profile.name} workload...")
        
        # Generate requests based on profile
        requests = self._generate_requests(workload_profile)
        results = []
        
        # Execute requests
        for request in requests:
            result = self._execute_request(graph, request, config)
            results.append(result)
            
        # Analyze results
        return self._analyze_results(workload_profile.name, results)
        
    def _generate_requests(self, profile: WorkloadProfile) -> List[WorkloadRequest]:
        """Generate requests based on workload profile."""
        requests = []
        start_time = time.time()
        current_time = start_time
        
        while current_time - start_time < profile.duration:
            # Check for burst
            if random.random() < profile.burst_probability:
                # Burst: generate multiple requests quickly
                burst_count = int(profile.arrival_rate * profile.burst_multiplier)
                for _ in range(burst_count):
                    request = profile.request_pattern()
                    request.timestamp = current_time
                    requests.append(request)
            else:
                # Normal rate
                if random.random() < profile.arrival_rate / 10:  # Convert to per-100ms
                    request = profile.request_pattern()
                    request.timestamp = current_time
                    requests.append(request)
                    
            current_time = time.time()
            time.sleep(0.1)  # 100ms intervals
            
        return requests
        
    def _execute_request(
        self, 
        graph: Pregel, 
        request: WorkloadRequest, 
        config: Optional[dict]
    ) -> WorkloadResult:
        """Execute a single request and measure performance."""
        start_time = time.time()
        success = True
        error_message = None
        tools_used = []
        
        try:
            # Execute graph
            if hasattr(graph, 'astream'):
                # Async execution
                asyncio.run(self._run_async(graph, request.input_data, config))
            else:
                # Sync execution
                list(graph.stream(request.input_data, config=config))
                
        except Exception as e:
            success = False
            error_message = str(e)
            
        execution_time = time.time() - start_time
        
        # Measure resource usage (simplified)
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024
            cpu_usage = process.cpu_percent()
        except ImportError:
            # Mock data if psutil not available
            memory_usage = 100.0
            cpu_usage = 50.0
        
        # Estimate response quality (simplified)
        response_quality = 1.0 if success else 0.0
        if success and execution_time < 5.0:  # Fast response
            response_quality = min(1.0, response_quality + 0.2)
            
        return WorkloadResult(
            request_id=request.request_id,
            execution_time=execution_time,
            success=success,
            error_message=error_message,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            tools_used=tools_used,
            response_quality=response_quality
        )
        
    async def _run_async(self, graph: Pregel, input_data: dict, config: Optional[dict]):
        """Run graph asynchronously."""
        async for _ in graph.astream(input_data, config=config):
            pass
            
    def _analyze_results(self, workload_name: str, results: List[WorkloadResult]) -> WorkloadReport:
        """Analyze execution results and generate report."""
        if not results:
            return WorkloadReport(
                workload_name=workload_name,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_execution_time=0,
                p95_execution_time=0,
                p99_execution_time=0,
                avg_throughput=0,
                peak_throughput=0,
                avg_memory_usage=0,
                peak_memory_usage=0,
                avg_cpu_usage=0,
                peak_cpu_usage=0,
                results=[],
                recommendations=[]
            )
            
        # Basic metrics
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r.success)
        failed_requests = total_requests - successful_requests
        
        # Execution time metrics
        execution_times = [r.execution_time for r in results if r.success]
        if execution_times:
            execution_times.sort()
            avg_execution_time = sum(execution_times) / len(execution_times)
            p95_execution_time = execution_times[int(len(execution_times) * 0.95)]
            p99_execution_time = execution_times[int(len(execution_times) * 0.99)]
        else:
            avg_execution_time = p95_execution_time = p99_execution_time = 0
            
        # Throughput metrics
        if execution_times:
            avg_throughput = len(execution_times) / sum(execution_times)
            peak_throughput = 1 / min(execution_times) if execution_times else 0
        else:
            avg_throughput = peak_throughput = 0
            
        # Resource usage metrics
        memory_usages = [r.memory_usage for r in results]
        cpu_usages = [r.cpu_usage for r in results]
        
        avg_memory_usage = sum(memory_usages) / len(memory_usages) if memory_usages else 0
        peak_memory_usage = max(memory_usages) if memory_usages else 0
        avg_cpu_usage = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
        peak_cpu_usage = max(cpu_usages) if cpu_usages else 0
        
        # Generate recommendations
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        recommendations = self._generate_recommendations(results, avg_execution_time, success_rate)
        
        return WorkloadReport(
            workload_name=workload_name,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_execution_time=avg_execution_time,
            p95_execution_time=p95_execution_time,
            p99_execution_time=p99_execution_time,
            avg_throughput=avg_throughput,
            peak_throughput=peak_throughput,
            avg_memory_usage=avg_memory_usage,
            peak_memory_usage=peak_memory_usage,
            avg_cpu_usage=avg_cpu_usage,
            peak_cpu_usage=peak_cpu_usage,
            results=results,
            recommendations=recommendations
        )
        
    def _generate_recommendations(
        self, 
        results: List[WorkloadResult], 
        avg_execution_time: float,
        success_rate: float
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if success_rate < 0.9:
            recommendations.append("High failure rate detected - investigate error handling")
            
        if avg_execution_time > 10:
            recommendations.append("Average execution time is high - consider optimization")
            
        # Check for memory issues
        memory_usages = [r.memory_usage for r in results]
        if memory_usages and max(memory_usages) > 1000:  # 1GB
            recommendations.append("High memory usage detected - consider memory optimization")
            
        # Check for CPU issues
        cpu_usages = [r.cpu_usage for r in results]
        if cpu_usages and max(cpu_usages) > 80:
            recommendations.append("High CPU usage detected - consider load balancing")
            
        return recommendations
