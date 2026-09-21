"""
Tool Registry and Execution Dispatcher for the Agentic Model.
Exposes tools for scratch generation, Ultra-HD diffusion, visual quality inspection,
and prompt refinement.
"""

from typing import Dict, Any, Callable, Optional, List
import inspect
from PIL import Image
import torch

from .schema import ToolCall, ToolResponse, CriticReport
from .critic import VisualCritic


class ToolRegistry:
    """
    Central registry for tools that the Agent can autonomously invoke.
    """

    def __init__(self, critic_threshold: float = 7.0):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self.critic = VisualCritic(aesthetic_threshold=critic_threshold)

        # Register default tools
        self._register_default_tools()

    def register(self, name: str, func: Callable, description: str, parameters: Dict[str, Any]):
        """Registers a tool with its callable handler and JSON schema."""
        self._tools[name] = func
        self._schemas[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Returns schemas for all registered tools."""
        return list(self._schemas.values())

    def execute(self, tool_call: ToolCall) -> ToolResponse:
        """Executes a ToolCall and returns a ToolResponse."""
        if tool_call.name not in self._tools:
            return ToolResponse(
                call_id=tool_call.call_id,
                name=tool_call.name,
                success=False,
                output=None,
                error=f"Tool '{tool_call.name}' not found in registry.",
            )

        handler = self._tools[tool_call.name]
        try:
            result = handler(**tool_call.arguments)
            return ToolResponse(
                call_id=tool_call.call_id,
                name=tool_call.name,
                success=True,
                output=result,
            )
        except Exception as e:
            return ToolResponse(
                call_id=tool_call.call_id,
                name=tool_call.name,
                success=False,
                output=None,
                error=f"Tool execution failed: {str(e)}",
            )

    def _register_default_tools(self):
        # 1. Image Quality Critic Tool
        def analyze_quality(image: Image.Image, prompt: str = "") -> CriticReport:
            return self.critic.evaluate(image, prompt=prompt)

        self.register(
            name="analyze_image_quality",
            func=analyze_quality,
            description="Analyzes sharpness, contrast, color harmony, and aesthetic score of an image.",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Original text prompt"}
                },
                "required": ["image"],
            },
        )

        # 2. Prompt Refiner Tool
        def refine_prompt(
            current_prompt: str,
            refinement_suggestions: List[str],
            style_boost: Optional[str] = None,
        ) -> str:
            parts = [current_prompt.strip()]
            if refinement_suggestions:
                # Add top 2 suggestions without duplicate words
                for sug in refinement_suggestions[:2]:
                    clean_sug = sug.replace("Inject '", "").replace("'", "")
                    if clean_sug.lower() not in current_prompt.lower():
                        parts.append(clean_sug)
            if style_boost and style_boost.lower() not in current_prompt.lower():
                parts.append(style_boost)
            return ", ".join(parts)

        self.register(
            name="refine_prompt",
            func=refine_prompt,
            description="Enriches and optimizes an image prompt using critic feedback and visual style boosters.",
            parameters={
                "type": "object",
                "properties": {
                    "current_prompt": {"type": "string"},
                    "refinement_suggestions": {"type": "array", "items": {"type": "string"}},
                    "style_boost": {"type": "string"},
                },
                "required": ["current_prompt"],
            },
        )
