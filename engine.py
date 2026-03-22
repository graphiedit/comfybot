"""
AI Director Engine — the central orchestrator.

Connects all components: LLM → Registry → WorkflowBuilder → ComfyUI → Discord.
Handles the full lifecycle: analyze prompt → build workflow → generate → respond.
"""
import asyncio
import dataclasses
import io
import logging
import tempfile
import os
from pathlib import Path
from typing import Optional

import discord

from llm.base import create_llm_provider, GenerationPlan
from registry.model_registry import ModelRegistry
from core.comfyui_client import ComfyUIClient
from core.workflow_builder import WorkflowBuilder
from core.queue_manager import QueueManager, Job, JobStatus
from core.pipeline import GenerationPipeline
from core.quality_analyzer import QualityScore
from core.user_memory import UserMemory
from core.image_analyzer import ImageAnalyzer
from core.cache import MemoryCache
from registry.style_presets import StylePresetManager
from discord_ui.embeds import (
    create_progress_embed,
    create_generation_embed,
    create_error_embed,
)
from discord_ui.buttons import GenerationButtons

logger = logging.getLogger(__name__)


class AIDirectorEngine:
    """
    The AI Director — orchestrates intelligent image generation.
    
    Connects LLM analysis, model selection, workflow building,
    ComfyUI execution, and Discord responses into a seamless pipeline.
    """

    def __init__(self, config: dict):
        self.config = config
        
        # Initialize components
        self.llm = create_llm_provider(config)
        self.registry = ModelRegistry(config)
        self.comfyui = ComfyUIClient(config)
        self.builder = WorkflowBuilder(config, registry=self.registry)
        self.queue = QueueManager(max_concurrent=1)
        
        # New advanced capabilities
        self.pipeline = GenerationPipeline(self, config)
        self.user_memory = UserMemory(config)
        self.style_presets = StylePresetManager()
        self.image_analyzer = ImageAnalyzer(self.llm)
        self.cache = MemoryCache(
            max_entries=config.get("cache", {}).get("max_entries", 100),
            default_ttl=config.get("cache", {}).get("ttl_seconds", 3600)
        )
        
        # Per-user conversation history for chat context
        self._chat_history: dict[str, list] = {}
        
        # Job metadata storage (plan details for buttons)
        self._job_metadata: dict[str, dict] = {}
        self._job_interactions: dict[str, discord.Interaction] = {}

    async def start(self):
        """Initialize all subsystems."""
        logger.info("Starting AI Director Engine...")
        
        # Load model registry
        await self.registry.refresh()
        
        # Start job queue
        self.queue.set_processor(self._process_job)
        await self.queue.start()
        
        # Check ComfyUI
        alive = await self.comfyui.is_alive()
        if alive:
            logger.info("✅ ComfyUI is running and reachable")
        else:
            logger.warning("⚠️ ComfyUI is not reachable — generation will fail until it's started")
        
        logger.info("AI Director Engine ready!")

    async def stop(self):
        """Shut down the engine."""
        await self.queue.stop()
        logger.info("AI Director Engine stopped")

    async def submit_generation(
        self,
        interaction: discord.Interaction,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        user_overrides: dict = None,
    ):
        """
        Submit an image generation request from Discord.
        
        This is the main entry point for /generate, /edit, /vary commands.
        """
        job = Job(
            user_id=str(interaction.user.id),
            guild_id=str(interaction.guild_id) if interaction.guild_id else "",
            channel_id=str(interaction.channel_id),
            prompt=prompt,
            metadata={
                "user_overrides": user_overrides or {},
                "has_image": image_bytes is not None,
            },
        )
        
        # Store image bytes temporarily
        if image_bytes:
            job.metadata["image_bytes"] = image_bytes
        
        # Store interaction for response
        self._job_interactions[job.id] = interaction
        
        # Set up status callback
        job.on_status_change = self._on_job_status_change
        
        # Send initial progress embed
        progress_embed = create_progress_embed(job, self.queue.get_queue_size() + 1)
        await interaction.followup.send(embed=progress_embed)
        
        # Submit to queue
        await self.queue.submit(job)

    async def submit_upscale(
        self,
        interaction: discord.Interaction,
        image_bytes: bytes,
        scale: int = 2,
    ):
        """Submit an upscale request."""
        job = Job(
            user_id=str(interaction.user.id),
            guild_id=str(interaction.guild_id) if interaction.guild_id else "",
            channel_id=str(interaction.channel_id),
            prompt="upscale",
            metadata={
                "user_overrides": {"action": "upscale"},
                "has_image": True,
                "image_bytes": image_bytes,
                "scale": scale,
            },
        )
        
        self._job_interactions[job.id] = interaction
        job.on_status_change = self._on_job_status_change
        
        progress_embed = create_progress_embed(job, self.queue.get_queue_size() + 1)
        await interaction.followup.send(embed=progress_embed)
        
        await self.queue.submit(job)

    async def handle_button_action(
        self,
        interaction: discord.Interaction,
        action: str,
        job_id: str,
    ):
        """Handle post-generation button presses."""
        original_meta = self._job_metadata.get(job_id, {})
        
        if action == "info":
            # Show detailed info about the generation
            info_lines = []
            for key, value in original_meta.items():
                if key not in ("image_bytes", "user_overrides"):
                    info_lines.append(f"**{key}**: {value}")
            info_text = "\n".join(info_lines) if info_lines else "No metadata available"
            await interaction.response.send_message(info_text[:2000], ephemeral=True)
            return
        
        await interaction.response.defer()
        
        if action == "retry":
            # Same prompt, new seed
            overrides = original_meta.get("user_overrides", {}).copy()
            overrides["seed"] = -1
            await self.submit_generation(
                interaction=interaction,
                prompt=original_meta.get("prompt", ""),
                user_overrides=overrides,
            )
        
        elif action == "upscale" and original_meta.get("result_images"):
            # Upscale the first result image
            await self.submit_upscale(
                interaction=interaction,
                image_bytes=original_meta["result_images"][0],
            )
        
        elif action in ("vary_subtle", "vary_strong"):
            denoise = 0.3 if action == "vary_subtle" else 0.7
            overrides = original_meta.get("user_overrides", {}).copy()
            overrides["denoise"] = denoise
            overrides["seed"] = -1
            await self.submit_generation(
                interaction=interaction,
                prompt=original_meta.get("prompt", ""),
                image_bytes=original_meta.get("result_images", [None])[0],
                user_overrides=overrides,
            )

    async def chat(self, user_id: str, message: str) -> str:
        """Chat with the AI Director."""
        history = self._chat_history.setdefault(user_id, [])
        
        # Build system context with available models
        available = self.registry.get_available_models_for_llm()
        ckpt_names = [c["filename"] if isinstance(c, dict) else c for c in available.get("checkpoints", [])]
        dm_names = [d["filename"] if isinstance(d, dict) else d for d in available.get("diffusion_models", [])]
        all_models = ckpt_names[:3] + dm_names[:3]
        
        context = (
            "You are an AI art assistant for a Stable Diffusion image generation system.\n"
            f"Available models: {', '.join(all_models)}\n"
            "You can help users craft prompts, explain models/LoRAs, and suggest settings.\n"
            "If they want to generate, tell them to use /generate command."
        )
        
        response = await self.llm.chat(message, history, system_context=context)
        
        # Update conversation history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        # Keep history manageable
        if len(history) > 20:
            history[:] = history[-20:]
        
        return response

    async def _process_job(self, job: Job) -> list:
        """
        Process a single generation job through the full pipeline.
        
        This is called by the QueueManager for each job.
        Returns a list of image bytes.
        """
        overrides = job.metadata.get("user_overrides", {})
        action = overrides.get("action", "generate")
        
        # --- Step 1: Upload reference image if provided ---
        reference_remote = None
        pose_remote = None
        
        if job.metadata.get("image_bytes"):
            job.status = JobStatus.BUILDING
            await self._notify_discord(job)
            
            # Save to temp file and upload to ComfyUI
            tmp_path = os.path.join(tempfile.gettempdir(), f"ai_director_{job.id}.png")
            with open(tmp_path, "wb") as f:
                f.write(job.metadata["image_bytes"])
            
            try:
                remote_name = await self.comfyui.upload_image(tmp_path)
                
                if action == "upscale":
                    reference_remote = remote_name
                else:
                    # Let Image Analyzer decide the best tools for reference
                    analysis = await self.image_analyzer.analyze(job.metadata["image_bytes"])
                    logger.info(f"Image Analysis: {analysis.raw_response}")
                    
                    if "ipadapter" in analysis.recommended_tools or analysis.style_info:
                        reference_remote = remote_name
                        
                    if "openpose" in analysis.recommended_tools and analysis.has_pose:
                        pose_remote = remote_name
                    elif "depth" in analysis.recommended_tools and analysis.has_depth:
                        pose_remote = remote_name # Using depth as "pose" structurally
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        
        # --- Step 2: Analyze intent with LLM ---
        job.status = JobStatus.ANALYZING
        await self._notify_discord(job)
        
        available_models = self.registry.get_available_models_for_llm()
        user_prefs = self.user_memory.get_preferences_for_llm(job.user_id)
        
        if action == "upscale":
            # Simple upscale plan
            plan = GenerationPlan(
                action="upscale",
                checkpoint=overrides.get("checkpoint", ""),
                enhanced_prompt="masterpiece, best quality, highres, 8k uhd, detailed",
                width=overrides.get("width", 1024),
                height=overrides.get("height", 1024),
            )
        else:
            # Check cache first
            cache_key = self.cache.generate_prompt_key(job.prompt, overrides)
            cached_plan_dict = self.cache.get(cache_key) if self.config.get("cache", {}).get("enabled", True) else None
            
            if cached_plan_dict:
                logger.info(f"Cache hit for prompt: {job.prompt}")
                plan = GenerationPlan(**cached_plan_dict)
            else:
                # Add user context to analysis
                full_prompt = f"{job.prompt}\n\n[USER PREFERENCES Context]\n{user_prefs}"
                plan = await self.llm.analyze_intent(
                    prompt=full_prompt,
                    available_models=available_models,
                    has_reference_image=job.metadata.get("has_image", False),
                )
                
                # Apply style preset if matched
                if plan.style_category:
                    preset = self.style_presets.get_preset(plan.style_category)
                    if preset:
                        plan = self.style_presets.apply_preset(plan, preset)
                
                if self.config.get("cache", {}).get("enabled", True):
                    self.cache.set(cache_key, dataclasses.asdict(plan))
        # Apply user overrides — user choices take priority over AI
        for key, value in overrides.items():
            if hasattr(plan, key) and key != "action":
                setattr(plan, key, value)
        
        # Validate model choices against registry
        if plan.checkpoint and not self.registry.validate_checkpoint(plan.checkpoint):
            logger.warning(f"AI chose unavailable checkpoint: {plan.checkpoint}, using default")
            plan.checkpoint = self.registry.get_best_checkpoint(plan.style_category) or ""
        
        validated_loras = []
        for lora in plan.loras:
            if self.registry.validate_lora(lora.get("name", "")):
                validated_loras.append(lora)
            else:
                logger.warning(f"AI chose unavailable LoRA: {lora.get('name')}, skipping")
        plan.loras = validated_loras
        
        # Fill in defaults from registry
        if not plan.checkpoint:
            plan.checkpoint = self.registry.get_best_checkpoint(plan.style_category) or self.config.get("defaults", {}).get("ckpt", "Juggernaut-XL_v9_RunDiffusion.safetensors")
        
        # Detect architecture from model name
        plan.model_arch = self.registry.get_model_arch(plan.checkpoint)
        
        # Apply architecture-specific defaults if the LLM didn't set them correctly
        model_defaults = self.registry.get_model_defaults(plan.checkpoint)
        if model_defaults:
            # Only override if user/LLM didn't explicitly set these
            if plan.model_arch in ("flux", "hunyuan"):
                if plan.cfg > 5.0:  # LLM probably defaulted to SDXL CFG
                    plan.cfg = model_defaults.get("cfg", 1.0)
                if plan.steps > 25 and model_defaults.get("default_steps", 30) < 10:
                    plan.steps = model_defaults.get("steps", plan.steps)
                plan.sampler = model_defaults.get("sampler", plan.sampler)
                plan.scheduler = model_defaults.get("scheduler", plan.scheduler)
        
        logger.info(f"Plan: arch={plan.model_arch}, model={plan.checkpoint}, steps={plan.steps}, cfg={plan.cfg}")
        
        # --- Step 3-4: Build & Execute Pipeline ---
        async def on_status_update(status: JobStatus):
            job.status = status
            await self._notify_discord(job)
            
        result_images = await self.pipeline.execute(
            job.id, plan, reference_remote, pose_remote, on_status_update
        )
        
        if not result_images:
            raise RuntimeError("Pipeline returned no images")
            
        # Record successful generation to user memory
        if action not in ("upscale", "edit"):
            self.user_memory.record_generation(job.user_id, plan)
        
        # Store metadata for button actions
        plan_dict = dataclasses.asdict(plan)
        plan_dict["prompt"] = job.prompt
        plan_dict["result_images"] = result_images
        self._job_metadata[job.id] = plan_dict
        
        return result_images

    async def _on_job_status_change(self, job: Job):
        """Called whenever a job's status changes — updates Discord."""
        await self._notify_discord(job)

    async def _notify_discord(self, job: Job):
        """Send status updates and results to Discord."""
        interaction = self._job_interactions.get(job.id)
        if not interaction:
            return
        
        try:
            if job.status == JobStatus.COMPLETE and job.result_images:
                # Send result with image and buttons
                plan_meta = self._job_metadata.get(job.id, {})
                embed = create_generation_embed(job, plan_meta)
                
                # Create buttons
                buttons = GenerationButtons(
                    job_id=job.id,
                    on_action=self.handle_button_action,
                )
                
                # Send image as file
                image_file = discord.File(
                    io.BytesIO(job.result_images[0]),
                    filename=f"generation_{job.id}.png",
                )
                embed.set_image(url=f"attachment://generation_{job.id}.png")
                
                await interaction.channel.send(
                    embed=embed,
                    file=image_file,
                    view=buttons,
                )
                
                # Clean up interaction reference
                if job.id in self._job_interactions:
                    del self._job_interactions[job.id]
            
            elif job.status == JobStatus.FAILED:
                embed = create_error_embed(job.error, job)
                await interaction.channel.send(embed=embed)
                
                if job.id in self._job_interactions:
                    del self._job_interactions[job.id]
            
            else:
                # Progress updates — could edit original message
                # For now, just log
                logger.debug(f"Job {job.id}: {job.status.value}")
        
        except Exception as e:
            logger.error(f"Failed to notify Discord for job {job.id}: {e}")
