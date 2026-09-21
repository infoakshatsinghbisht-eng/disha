"""
Agentic Multimodal Image Generation Pipeline.
Connects Cognitive Reasoning Brain (<think>), Tool Execution Registry (<tool_call>),
Visual Quality Critic (<tool_response>), and Autonomous ReAct Reflection Loop.
"""

from typing import Optional, Callable, Dict, Any, Union, Tuple
import os
import torch
from PIL import Image

from agent.schema import AgentExecutionTrace, ThinkingStep, CriticReport
from agent.tools import ToolRegistry
from agent.reasoner import AgentReasoner
from agent.critic import VisualCritic
from config import AgenticConfig, GenerationConfig, LLMConfig, VQVAEConfig
from tokenizer.text_tokenizer import ByteTokenizer
from vqvae.model import VQVAE
from model.transformer import MultimodalTransformer
from .sampler import MultimodalGeneratorPipeline


class AgenticImagePipeline:
    """
    End-to-End Agentic Multimodal Image Generation Studio.
    Orchestrates Chain-of-Thought planning, multi-engine dispatch, and automated quality critique.
    """

    def __init__(
        self,
        scratch_pipeline: Optional[MultimodalGeneratorPipeline] = None,
        config: Optional[AgenticConfig] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        self.device = torch.device(device)
        self.config = config or AgenticConfig()
        self.scratch_pipeline = scratch_pipeline

        # Initialize Tool Registry with Critic
        self.registry = ToolRegistry(critic_threshold=self.config.aesthetic_threshold)
        
        # Register Engine 1 (Scratch Foundation Model)
        self._register_scratch_engine()
        
        # Register Engine 2 (Ultra-HD Diffusion Core)
        self._register_hd_engine()

        # Initialize Reasoner Brain
        self.reasoner = AgentReasoner(
            tool_registry=self.registry,
            aesthetic_threshold=self.config.aesthetic_threshold,
            max_refinement_steps=self.config.max_refinement_steps,
        )

    def _register_scratch_engine(self):
        def generate_scratch(prompt: str, **kwargs) -> Image.Image:
            if self.scratch_pipeline is not None:
                gen_cfg = GenerationConfig(
                    temperature=kwargs.get("temperature", 0.75),
                    top_k=kwargs.get("top_k", 40),
                )
                return self.scratch_pipeline.generate_image(prompt, gen_config=gen_cfg)
            else:
                # Synthetic fallback for testing without loaded weights
                img = Image.new("RGB", (256, 256), color=(30, 34, 42))
                return img

        self.registry.register(
            name="generate_image_scratch",
            func=generate_scratch,
            description="Synthesizes 256x256 discrete visual tokens via from-scratch PyTorch Multimodal LLM.",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "temperature": {"type": "number"},
                },
                "required": ["prompt"],
            },
        )

    def _register_hd_engine(self):
        def generate_hd(prompt: str, **kwargs) -> Image.Image:
            try:
                from generate_hd import generate_hd_image
                return generate_hd_image(
                    prompt=prompt,
                    num_inference_steps=kwargs.get("num_inference_steps", 2),
                    guidance_scale=kwargs.get("guidance_scale", 0.0),
                    device=str(self.device),
                )
            except Exception as e:
                print(f"[!] HD Engine warning ({e}). Falling back to Scratch Engine.")
                return self.registry.execute(self.registry.ToolCall(
                    name="generate_image_scratch", arguments={"prompt": prompt}
                )).output

        self.registry.register(
            name="generate_image_hd",
            func=generate_hd,
            description="Synthesizes 1024x1024 studio-grade image via Ultra-HD Latent Diffusion Core.",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "num_inference_steps": {"type": "integer"},
                },
                "required": ["prompt"],
            },
        )

    @classmethod
    def from_scratch_checkpoint(
        cls,
        checkpoint_path: str = "checkpoints/multimodal_llm.pt",
        device: Union[str, torch.device] = "cpu",
        config: Optional[AgenticConfig] = None,
        dim: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> "AgenticImagePipeline":
        """Loads scratch pipeline from checkpoint if available, or lightweight baseline."""
        scratch_pipe = None
        if os.path.exists(checkpoint_path):
            try:
                scratch_pipe = MultimodalGeneratorPipeline.from_pretrained(checkpoint_path, device=device)
            except Exception as e:
                print(f"[!] Could not load checkpoint ({e}), initializing baseline pipeline.")
        
        if scratch_pipe is None:
            llm_cfg = LLMConfig()
            vq_cfg = VQVAEConfig()
            tokenizer = ByteTokenizer()
            total_vocab = llm_cfg.text_vocab_size + llm_cfg.image_vocab_size + len(tokenizer.SPECIAL_TOKENS)

            is_cpu = str(device).startswith("cpu")
            model_dim = dim if dim is not None else (128 if is_cpu else llm_cfg.dim)
            layers = num_layers if num_layers is not None else (4 if is_cpu else llm_cfg.num_layers)
            heads = 4 if model_dim < 512 else llm_cfg.num_heads
            kv_heads = 2 if model_dim < 512 else llm_cfg.num_kv_heads

            llm = MultimodalTransformer(
                vocab_size=total_vocab,
                dim=model_dim,
                num_layers=layers,
                num_heads=heads,
                num_kv_heads=kv_heads,
                max_seq_len=llm_cfg.max_seq_len,
            )
            vqvae = VQVAE(
                in_channels=vq_cfg.in_channels,
                hidden_dim=32 if model_dim < 512 else vq_cfg.hidden_dim,
                embedding_dim=16 if model_dim < 512 else vq_cfg.embedding_dim,
                codebook_size=vq_cfg.codebook_size,
                num_res_blocks=1 if model_dim < 512 else vq_cfg.num_res_blocks,
                num_downsamples=vq_cfg.num_downsamples,
            )
            scratch_pipe = MultimodalGeneratorPipeline(
                llm=llm,
                vqvae=vqvae,
                tokenizer=tokenizer,
                text_vocab_size=llm_cfg.text_vocab_size,
                image_token_len=llm_cfg.image_token_len,
                device=device,
            )

        return cls(scratch_pipeline=scratch_pipe, config=config, device=device)

    def run(
        self,
        prompt: str,
        engine: str = "scratch",
        style: Optional[str] = None,
        auto_refine: bool = True,
        on_step_callback: Optional[Callable[[ThinkingStep], None]] = None,
    ) -> AgentExecutionTrace:
        """
        Executes the full ReAct agentic cycle for a given user prompt.
        """
        def generate_callback(p: str, args: Dict[str, Any]) -> Image.Image:
            selected_tool = "generate_image_scratch" if engine == "scratch" else "generate_image_hd"
            from agent.schema import ToolCall
            resp = self.registry.execute(ToolCall(name=selected_tool, arguments={"prompt": p}))
            if resp.success and isinstance(resp.output, Image.Image):
                return resp.output
            # Fallback
            return Image.new("RGB", (256, 256), color=(35, 39, 46))

        trace = self.reasoner.run_react_cycle(
            prompt=prompt,
            engine=engine,
            generate_fn=generate_callback,
            style=style,
            auto_refine=auto_refine,
            on_step_callback=on_step_callback,
        )
        return trace
