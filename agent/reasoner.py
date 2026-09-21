"""
Chain-of-Thought (<think>) Reasoner and Autonomous ReAct Loop Controller.
Performs semantic decomposition, prompt synthesis, tool calling, and automated visual critique
with multi-step reflection and self-correction.
"""

from typing import Optional, Callable, Dict, Any, List
import time
from PIL import Image

from .schema import (
    ToolCall,
    ToolResponse,
    CriticReport,
    ThinkingStep,
    AgentExecutionTrace,
)
from .tools import ToolRegistry
from .critic import VisualCritic


class AgentReasoner:
    """
    Cognitive Agent Brain for Multimodal Image Generation.
    Executes Chain-of-Thought planning and autonomous ReAct reflection loops.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        aesthetic_threshold: float = 7.0,
        max_refinement_steps: int = 2,
    ):
        self.registry = tool_registry
        self.aesthetic_threshold = aesthetic_threshold
        self.max_refinement_steps = max_refinement_steps

    def _think_analysis(self, prompt: str, engine: str) -> str:
        """Simulates internal Chain-of-Thought semantic intent analysis."""
        prompt_lower = prompt.lower()
        
        # Detect primary genre / theme
        genre = "Photorealistic Fine-Art"
        if any(w in prompt_lower for w in ["cyberpunk", "sci-fi", "robot", "neon", "future"]):
            genre = "Cyberpunk / Sci-Fi"
        elif any(w in prompt_lower for w in ["anime", "ghibli", "manga", "cartoon"]):
            genre = "Anime / Studio Ghibli"
        elif any(w in prompt_lower for w in ["dark", "fantasy", "dragon", "magic", "dungeon"]):
            genre = "Dark Fantasy"
        elif any(w in prompt_lower for w in ["3d", "render", "octane", "sculpture"]):
            genre = "3D Octane Render"

        # Determine lighting & composition recommendations
        composition = "Rule-of-thirds balanced framing with volumetric depth"
        lighting = "Natural directional lighting with soft ambient occlusion"
        if "cyberpunk" in genre.lower():
            lighting = "Bioluminescent neon glow with wet reflections and high contrast"
        elif "dark" in genre.lower():
            lighting = "Moody chiaroscuro rim lighting with volumetric haze"

        thought = (
            f"User Prompt: '{prompt}'\n"
            f"1. Semantic Classification: Detected genre '{genre}'.\n"
            f"2. Framing & Composition: {composition}.\n"
            f"3. Lighting Scheme: {lighting}.\n"
            f"4. Selected Engine: '{engine}' based on resolution & execution target.\n"
            f"5. Action Plan: Synthesize enriched generation prompt and dispatch initial synthesis tool."
        )
        return thought

    def run_react_cycle(
        self,
        prompt: str,
        engine: str = "scratch",
        generate_fn: Optional[Callable[[str, Dict[str, Any]], Image.Image]] = None,
        style: Optional[str] = None,
        auto_refine: bool = True,
        on_step_callback: Optional[Callable[[ThinkingStep], None]] = None,
    ) -> AgentExecutionTrace:
        """
        Executes the full ReAct agent cycle:
        Reason (<think>) -> Act (<tool_call>) -> Observe (<tool_response>) -> Critique & Reflect.
        """
        start_time = time.time()
        trace = AgentExecutionTrace(
            user_prompt=prompt,
            engine_used=engine,
        )

        current_prompt = prompt
        current_image = None
        step_idx = 1

        # --- STEP 1: INITIAL ANALYSIS & COGNITIVE PLANNING ---
        analysis_thought = self._think_analysis(prompt, engine)
        step1 = ThinkingStep(
            step_number=step_idx,
            phase="ANALYSIS",
            thought=f"<think>\n{analysis_thought}\n</think>",
        )
        trace.thinking_steps.append(step1)
        if on_step_callback:
            on_step_callback(step1)
        step_idx += 1

        # Generate enriched prompt for initial synthesis
        from pipeline.prompt_enhancer import PromptEnhancer
        enriched_prompt = PromptEnhancer.enhance(prompt, style=style, add_quality_boosters=True)
        current_prompt = enriched_prompt

        iteration = 0
        max_loops = (self.max_refinement_steps + 1) if auto_refine else 1

        while iteration < max_loops:
            iteration += 1

            # --- STEP 2: GENERATION TOOL CALL ---
            gen_tool_name = "generate_image_scratch" if engine == "scratch" else "generate_image_hd"
            tool_args = {"prompt": current_prompt, "engine": engine}
            
            tool_call = ToolCall(name=gen_tool_name, arguments=tool_args)
            gen_thought = (
                f"<think>\n"
                f"Dispatching generation tool '{gen_tool_name}' for iteration {iteration}.\n"
                f"Active Prompt: \"{current_prompt}\"\n"
                f"</think>\n"
                f"<tool_call: {gen_tool_name} {{\"prompt\": \"{current_prompt}\", \"engine\": \"{engine}\"}}>"
            )

            step_gen = ThinkingStep(
                step_number=step_idx,
                phase="GENERATION",
                thought=gen_thought,
                tool_call=tool_call,
            )
            step_idx += 1

            # Execute image generation
            if generate_fn is not None:
                current_image = generate_fn(current_prompt, tool_args)
            else:
                # Fallback synthetic image if no generator hooked
                current_image = Image.new("RGB", (256, 256), color=(40, 44, 52))

            tool_resp = ToolResponse(
                call_id=tool_call.call_id,
                name=gen_tool_name,
                success=True,
                output=f"Image generated ({current_image.size[0]}x{current_image.size[1]} RGB)",
            )
            step_gen.tool_response = tool_resp
            trace.thinking_steps.append(step_gen)
            if on_step_callback:
                on_step_callback(step_gen)

            # --- STEP 3: VISUAL CRITIQUE ---
            critic_call = ToolCall(name="analyze_image_quality", arguments={"image": current_image, "prompt": current_prompt})
            critic_resp = self.registry.execute(critic_call)
            critic_report: CriticReport = critic_resp.output
            trace.critic_reports.append(critic_report)

            critique_thought = (
                f"<think>\n"
                f"Visual Critic Inspection Results (Iteration {iteration}):\n"
                f"- Aesthetic Score: {critic_report.aesthetic_score}/10.0 (Threshold: {self.aesthetic_threshold})\n"
                f"- Edge Sharpness: {critic_report.sharpness:.1f} Laplacian variance\n"
                f"- Contrast: {critic_report.contrast:.1f} RMS luminance std\n"
                f"- Color Richness: {critic_report.color_richness:.1f}/10\n"
                f"- Assessment: {critic_report.critique}\n"
                f"</think>\n"
                f"<tool_response: analyze_image_quality status=\"{critic_report.status}\" score={critic_report.aesthetic_score}>"
            )

            step_critique = ThinkingStep(
                step_number=step_idx,
                phase="CRITIQUE",
                thought=critique_thought,
                tool_call=critic_call,
                tool_response=critic_resp,
            )
            trace.thinking_steps.append(step_critique)
            if on_step_callback:
                on_step_callback(step_critique)
            step_idx += 1

            # --- STEP 4: REFLECTION & SELF-CORRECTION DECISION ---
            if critic_report.status == "APPROVED" or iteration >= max_loops:
                conclusion_thought = (
                    f"<think>\n"
                    f"Decision: Image quality meets acceptance criteria "
                    f"({critic_report.aesthetic_score} >= {self.aesthetic_threshold} or max loops reached).\n"
                    f"Autonomous ReAct cycle completed successfully after {iteration} iteration(s).\n"
                    f"</think>"
                )
                step_conclusion = ThinkingStep(
                    step_number=step_idx,
                    phase="CONCLUSION",
                    thought=conclusion_thought,
                )
                trace.thinking_steps.append(step_conclusion)
                if on_step_callback:
                    on_step_callback(step_conclusion)
                break
            else:
                # Needs refinement
                refine_call = ToolCall(
                    name="refine_prompt",
                    arguments={
                        "current_prompt": current_prompt,
                        "refinement_suggestions": critic_report.suggested_refinements,
                    },
                )
                refine_resp = self.registry.execute(refine_call)
                refined_prompt = refine_resp.output

                reflection_thought = (
                    f"<think>\n"
                    f"Reflection & Self-Correction Plan:\n"
                    f"The visual critic identified flaws: {critic_report.critique}\n"
                    f"Action: Refining generation prompt with targeted micro-texture and lighting boosters.\n"
                    f"New Prompt: \"{refined_prompt}\"\n"
                    f"Preparing Iteration {iteration + 1}...\n"
                    f"</think>"
                )
                step_reflect = ThinkingStep(
                    step_number=step_idx,
                    phase="REFLECTION",
                    thought=reflection_thought,
                    tool_call=refine_call,
                    tool_response=refine_resp,
                )
                trace.thinking_steps.append(step_reflect)
                if on_step_callback:
                    on_step_callback(step_reflect)
                step_idx += 1

                current_prompt = refined_prompt

        trace.final_prompt = current_prompt
        trace.final_image = current_image
        trace.total_iterations = iteration
        trace.execution_time_sec = round(time.time() - start_time, 2)
        return trace
