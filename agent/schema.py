"""
Data Schemas for the Agentic Multimodal Image Generation System.
Defines Tool Calls, Tool Responses, Visual Critic Reports, and ReAct Execution Traces.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time


@dataclass
class ToolCall:
    """Represents a structured tool invocation requested by the Agent."""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: f"call_{int(time.time()*1000)%1000000}")


@dataclass
class ToolResponse:
    """Represents the observation returned from tool execution."""
    call_id: str
    name: str
    success: bool
    output: Any
    error: Optional[str] = None


@dataclass
class CriticReport:
    """Quantitative and qualitative assessment of generated image quality."""
    sharpness: float               # Laplacian variance (higher = sharper, blur < 100)
    contrast: float                # Standard deviation of luminance (0-100)
    dynamic_range: float           # Percentile span (0-255)
    color_richness: float          # Histogram entropy / color variety (0-10)
    aesthetic_score: float         # Aggregated aesthetic score (0.0 to 10.0)
    status: str                    # "APPROVED" or "NEEDS_REFINEMENT"
    critique: str                  # Descriptive critique explaining the score
    suggested_refinements: List[str] = field(default_factory=list)


@dataclass
class ThinkingStep:
    """Individual Chain-of-Thought (<think>) step in the ReAct loop."""
    step_number: int
    phase: str                     # "ANALYSIS", "PLANNING", "GENERATION", "CRITIQUE", "REFLECTION"
    thought: str
    tool_call: Optional[ToolCall] = None
    tool_response: Optional[ToolResponse] = None


@dataclass
class AgentExecutionTrace:
    """Complete audit trail of the agentic generation process."""
    user_prompt: str
    thinking_steps: List[ThinkingStep] = field(default_factory=list)
    final_prompt: str = ""
    engine_used: str = ""
    critic_reports: List[CriticReport] = field(default_factory=list)
    total_iterations: int = 1
    execution_time_sec: float = 0.0
    final_image: Any = None
