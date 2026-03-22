"""
AI Director Engine — the central orchestrator.

Connects all components: LLM → Registry → WorkflowBuilder → ComfyUI → Discord.
Handles the full lifecycle: analyze prompt → build workflow → generate → respond.

Now fully agentic:
  - Self-healing error recovery (auto-diagnoses and fixes ComfyUI errors)
  - LLM provider failover (auto-switches on rate limits)
  - Pre-generation intelligence (clarifications, suggestions)
  - Post-generation quality review
"""
import asyncio
import dataclasses
import io
import logging
import re
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

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
from core.error_recovery import ErrorDiagnosticAgent
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
        self.llm = create_llm_provider(config)  # Returns ProviderManager with failover
        self.registry = ModelRegistry(config)
        self.comfyui = ComfyUIClient(config)
        self.builder = WorkflowBuilder(config, registry=self.registry)
        
        from core.workflow_manager import WorkflowManager
        self.workflow_manager = WorkflowManager(config.get("data_dir", "data"))
        
        self.queue = QueueManager(max_concurrent=1)
        
        # Agentic capabilities
        self.error_agent = ErrorDiagnosticAgent(registry=self.registry)
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
        self._job_channels: dict[str, discord.abc.Messageable] = {}

    async def start(self):
        """Initialize all subsystems."""
        logger.info("Starting AI Director Engine...")
        
        # Initialize LLM provider(s) with failover
        await self.llm.initialize()
        logger.info(f"LLM providers ready: {self.llm.active_provider_name}")
        
        # Load model registry
        await self.registry.refresh()
        
        # Update error agent with fresh registry
        self.error_agent.registry = self.registry
        
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
        """Submit an image generation request interactively."""
        import uuid
        job_id = str(uuid.uuid4())
        
        await interaction.followup.send("💭 AI Director is drafting the perfect generation plan...", ephemeral=True)
        
        overrides = user_overrides or {}
        overrides["action"] = "generate"
        if image_bytes:
            overrides["has_image"] = True
            
        try:
            available_models = self.registry.get_available_models_for_llm()
            user_prefs = self.user_memory.get_preferences_for_llm(str(interaction.user.id))
            available_templates = self.workflow_manager.get_available_templates_for_llm()
            full_prompt = f"{prompt}\n\n[USER PREFERENCES Context]\n{user_prefs}\n{available_templates}"
            
            plan = await self.llm.analyze_intent(
                prompt=full_prompt,
                available_models=available_models,
                has_reference_image=image_bytes is not None,
            )
            
            if plan.style_category:
                preset = self.style_presets.get_preset(plan.style_category)
                if preset:
                    plan = self.style_presets.apply_preset(plan, preset)
            
            for key, value in overrides.items():
                if hasattr(plan, key) and key != "action":
                    setattr(plan, key, value)
                    
            if plan.checkpoint:
                resolved = self.registry.resolve_checkpoint(plan.checkpoint)
                plan.checkpoint = resolved if resolved else (self.registry.get_best_checkpoint(plan.style_category) or "")
            
            if not plan.checkpoint:
                plan.checkpoint = self.registry.get_best_checkpoint(plan.style_category) or self.config.get("defaults", {}).get("ckpt", "Juggernaut-XL_v9_RunDiffusion.safetensors")
            
            plan.model_arch = self.registry.get_model_arch(plan.checkpoint)
            
            # Interactive Conflict Resolution (Inverted)
            if plan.model_arch != "sdxl" and plan.use_ipadapter:
                plan.use_ipadapter = False
                logger.info(f"Dropped IPAdapter as model {plan.checkpoint} is {plan.model_arch}")
            if plan.model_arch != "sdxl" and plan.use_controlnet:
                plan.use_controlnet = False
                logger.info(f"Dropped ControlNet as model {plan.checkpoint} is {plan.model_arch}")
            
            # --- Interactive LoRA filtering (Compatibility Check) ---
            final_loras = []
            for lora in plan.loras:
                lora_name = lora.get("name", "")
                compat = self.registry.get_lora_compatibility(lora_name)
                if plan.model_arch in compat:
                    final_loras.append(lora)
                else:
                    logger.warning(f"In interactive flow, removing incompatible LoRA {lora_name} for model arch {plan.model_arch}")
            plan.loras = final_loras
                
            review = await self.llm.check_plan_completeness(plan, available_models)
            
            if plan.model_arch != "sdxl" and image_bytes:
                review.warnings.append(
                    f"Requested non-SDXL model `{plan.model_arch}` ignores Face Fix / Pose Match. "
                    "I have dropped these structural tools to let you test this model. If you want Face Fix, hit Reroll and ask for SDXL."
                )

            from discord_ui.embeds import create_plan_approval_embed
            from discord_ui.buttons import PlanApprovalView
            
            embed = create_plan_approval_embed(plan, review)
            
            async def on_approve(btn_interaction):
                job = Job(
                    user_id=str(interaction.user.id),
                    guild_id=str(interaction.guild_id) if interaction.guild_id else "",
                    channel_id=str(interaction.channel_id),
                    prompt=prompt,
                    metadata={
                        "user_overrides": overrides,
                        "has_image": image_bytes is not None,
                        "image_bytes": image_bytes,
                        "plan": plan, 
                    },
                )
                job.id = job_id
                self._job_interactions[job.id] = interaction
                job.on_status_change = self._on_job_status_change
                
                await self.queue.submit(job)
                progress_embed = create_progress_embed(job, self.queue.get_queue_size())
                await btn_interaction.channel.send(f"<@{interaction.user.id}>", embed=progress_embed)
                
            async def on_reroll(btn_interaction):
                import random
                overrides["seed"] = random.randint(1, 1000000)
                await btn_interaction.followup.send("🔄 Rerolling...", ephemeral=True)
                await self.submit_generation(interaction, prompt, image_bytes, overrides)
                
            async def on_cancel(btn_interaction):
                await btn_interaction.followup.send("❌ Cancelled.", ephemeral=True)
                
            view = PlanApprovalView(on_approve, on_reroll, on_cancel)
            await interaction.followup.send(embed=embed, view=view)
            
        except Exception as e:
            logger.error(f"Plan draft failed: {e}", exc_info=True)
            from discord_ui.embeds import create_error_embed
            err_emb = create_error_embed(f"Drafting plan failed: {str(e)}")
            await interaction.followup.send(embed=err_emb)

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

    async def chat(self, user_id: str, message: str, channel: Optional[discord.abc.Messageable] = None) -> str:
        """Chat with the AI Director."""
        history = self._chat_history.setdefault(user_id, [])
        
        # Build system context with available models and templates
        available = self.registry.get_available_models_for_llm()
        available_templates = self.workflow_manager.get_available_templates_for_llm()
        
        ckpt_names = [c["filename"] if isinstance(c, dict) else c for c in available.get("checkpoints", [])]
        dm_names = [d["filename"] if isinstance(d, dict) else d for d in available.get("diffusion_models", [])]
        all_models = ckpt_names[:5] + dm_names[:5]
        
        context = (
            "You are an AI art assistant for a Stable Diffusion image generation system.\n"
            f"Available models: {', '.join(all_models)}\n"
            f"{available_templates}\n"
            "You can help users craft prompts, explain models/LoRAs, and suggest settings.\n"
        )
        
        response = await self.llm.chat(message, history, system_context=context)
        
        # Check for autonomous generation tag: <generate>prompt</generate>
        gen_match = re.search(r"<generate>(.*?)</generate>", response, re.DOTALL | re.IGNORECASE)
        if gen_match and channel:
            gen_prompt = gen_match.group(1).strip()
            # Trigger generation in the background
            asyncio.create_task(self.submit_generation_from_chat(channel, user_id, gen_prompt))
            # Strip the tag from the response for the user
            response = response.replace(gen_match.group(0), "").strip()
            if not response:
                response = f"Rocketing away to generate: **{gen_prompt}**"
        
        # Update conversation history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        # Keep history manageable
        if len(history) > 20:
            history[:] = history[-20:]
        
        return response

    async def submit_generation_from_chat(
        self,
        channel: discord.abc.Messageable,
        user_id: str,
        prompt: str,
    ):
        """Submit generation triggered from a chat message (no interaction)."""
        import uuid
        job_id = str(uuid.uuid4())
        
        await channel.send(f"🎨 **AI Director Agent** is starting generation: *{prompt}*")
        
        try:
            available_models = self.registry.get_available_models_for_llm()
            user_prefs = self.user_memory.get_preferences_for_llm(user_id)
            available_templates = self.workflow_manager.get_available_templates_for_llm()
            full_prompt = f"{prompt}\n\n[USER PREFERENCES Context]\n{user_prefs}\n{available_templates}"
            
            plan = await self.llm.analyze_intent(
                prompt=full_prompt,
                available_models=available_models,
                has_reference_image=False,
            )
            
            # Apply same logic as submit_generation
            if plan.style_category:
                preset = self.style_presets.get_preset(plan.style_category)
                if preset:
                    plan = self.style_presets.apply_preset(plan, preset)
            
            if plan.checkpoint:
                resolved = self.registry.resolve_checkpoint(plan.checkpoint)
                plan.checkpoint = resolved if resolved else (self.registry.get_best_checkpoint(plan.style_category) or "")
            
            if not plan.checkpoint:
                plan.checkpoint = self.registry.get_best_checkpoint(plan.style_category) or self.config.get("defaults", {}).get("ckpt", "Juggernaut-XL_v9_RunDiffusion.safetensors")
            
            plan.model_arch = self.registry.get_model_arch(plan.checkpoint)
            
            # Compatibility filtering
            final_loras = []
            for lora in plan.loras:
                lora_name = lora.get("name", "")
                compat = self.registry.get_lora_compatibility(lora_name)
                if plan.model_arch in compat:
                    final_loras.append(lora)
            plan.loras = final_loras

            # Create the Job
            job = Job(
                user_id=user_id,
                guild_id="", # Hard to get from channel easily without interaction sometimes, but we can try
                channel_id=str(channel.id) if hasattr(channel, 'id') else "",
                prompt=prompt,
                metadata={
                    "plan": plan,
                    "from_chat": True
                },
            )
            job.id = job_id
            self._job_channels[job.id] = channel
            job.on_status_change = self._on_job_status_change
            
            await self.queue.submit(job)
            
        except Exception as e:
            logger.error(f"Chat-triggered generation failed: {e}")
            await channel.send(f"❌ Failed to start generation: {str(e)}")

    async def _process_job(self, job: Job) -> list:
        """
        Process a single generation job through the full agentic pipeline.
        
        This is called by the QueueManager for each job.
        Features:
          - Self-healing error recovery with automatic retries
          - Pre-generation intelligence (plan review)
          - Post-generation quality review  
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
        
        # --- Step 2: Analyze intent with LLM (if not provided interactively) ---
        plan = job.metadata.get("plan")
        
        if not plan:
            job.status = JobStatus.ANALYZING
            await self._notify_discord(job)
            
            available_models = self.registry.get_available_models_for_llm()
            user_prefs = self.user_memory.get_preferences_for_llm(job.user_id)
            
            if action == "upscale":
                plan = GenerationPlan(
                    action="upscale",
                    checkpoint=overrides.get("checkpoint", ""),
                    enhanced_prompt="masterpiece, best quality, highres, 8k uhd, detailed",
                    width=overrides.get("width", 1024),
                    height=overrides.get("height", 1024),
                )
            else:
                cache_key = self.cache.generate_prompt_key(job.prompt, overrides)
                cached_plan_dict = self.cache.get(cache_key) if self.config.get("cache", {}).get("enabled", True) else None
                
                if cached_plan_dict:
                    logger.info(f"Cache hit for prompt: {job.prompt}")
                    plan = GenerationPlan(**cached_plan_dict)
                else:
                    full_prompt = f"{job.prompt}\n\n[USER PREFERENCES Context]\n{user_prefs}"
                    try:
                        plan = await self.llm.analyze_intent(
                            prompt=full_prompt,
                            available_models=available_models,
                            has_reference_image=job.metadata.get("has_image", False),
                        )
                    except Exception as e:
                        logger.warning(f"All LLMs failed intent analysis: {e}. Using default plan.")
                        plan = GenerationPlan(
                            enhanced_prompt=job.prompt,
                            reasoning="Fallback: LLM brain unavailable, using defaults.",
                            action="generate"
                        )
                    
                    if plan.style_category:
                        preset = self.style_presets.get_preset(plan.style_category)
                        if preset:
                            plan = self.style_presets.apply_preset(plan, preset)
                    
                    if self.config.get("cache", {}).get("enabled", True):
                        self.cache.set(cache_key, dataclasses.asdict(plan))
            
            for key, value in overrides.items():
                if hasattr(plan, key) and key != "action":
                    setattr(plan, key, value)
            
            if plan.checkpoint:
                resolved = self.registry.resolve_checkpoint(plan.checkpoint)
                plan.checkpoint = resolved if resolved else (self.registry.get_best_checkpoint(plan.style_category) or "")
            
            validated_loras = []
            for lora in plan.loras:
                if self.registry.validate_lora(lora.get("name", "")):
                    validated_loras.append(lora)
            plan.loras = validated_loras
            
            if not plan.checkpoint:
                plan.checkpoint = self.registry.get_best_checkpoint(plan.style_category) or self.config.get("defaults", {}).get("ckpt", "Juggernaut-XL_v9_RunDiffusion.safetensors")
            
            plan.model_arch = self.registry.get_model_arch(plan.checkpoint)
            
            # --- Last-mile LoRA filtering (Compatibility Check) ---
            final_loras = []
            for lora in validated_loras:
                lora_name = lora.get("name", "")
                compat = self.registry.get_lora_compatibility(lora_name)
                if plan.model_arch in compat:
                    final_loras.append(lora)
                else:
                    logger.warning(f"Removing incompatible LoRA {lora_name} for model arch {plan.model_arch}")
            plan.loras = final_loras
            
            model_defaults = self.registry.get_model_defaults(plan.checkpoint)
            if model_defaults:
                if plan.model_arch in ("flux", "hunyuan"):
                    if plan.cfg > 5.0: plan.cfg = model_defaults.get("cfg", 1.0)
                    if plan.steps > 25 and model_defaults.get("default_steps", 30) < 10: plan.steps = model_defaults.get("steps", plan.steps)
                    plan.sampler = model_defaults.get("sampler", plan.sampler)
                    plan.scheduler = model_defaults.get("scheduler", plan.scheduler)
            
            # Legacy non-interactive conflict fallback (Inverted)
            if plan.model_arch != "sdxl" and plan.use_ipadapter:
                plan.use_ipadapter = False
                logger.info(f"Dropped IPAdapter as model {plan.checkpoint} is {plan.model_arch}")
            if plan.model_arch != "sdxl" and plan.use_controlnet:
                plan.use_controlnet = False
                logger.info(f"Dropped ControlNet as model {plan.checkpoint} is {plan.model_arch}")

        # At this point we definitely have a compiled plan
        logger.info(f"Plan Executing: arch={plan.model_arch}, model={plan.checkpoint}, steps={plan.steps}, cfg={plan.cfg}")
        
        # --- Step 4: Build & Execute with Self-Healing ---
        async def on_status_update(status):
            job.status = status
            await self._notify_discord(job)
        
        result_images = await self._execute_with_recovery(
            job, plan, reference_remote, pose_remote, on_status_update
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
        
        # Clear error recovery history
        self.error_agent.clear_history(job.id)
        
        return result_images

    async def _execute_with_recovery(
        self, job, plan, reference_remote, pose_remote, status_callback, max_retries=3
    ) -> list:
        """
        Execute the generation pipeline with automatic error recovery.
        
        If ComfyUI returns an error, the ErrorDiagnosticAgent diagnoses it
        and applies fixes (different model, lower resolution, SDXL fallback, etc.).
        """
        for attempt in range(max_retries + 1):
            try:
                result_images = await self.pipeline.execute(
                    job.id, plan, reference_remote, pose_remote, status_callback
                )
                return result_images
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Generation attempt {attempt + 1} failed: {error_msg[:200]}")
                
                if attempt >= max_retries:
                    raise
                
                # --- Agentic Error Recovery ---
                strategy = self.error_agent.diagnose(
                    error_msg, {}, plan, job_id=job.id
                )
                
                if not strategy.can_fix:
                    logger.error(f"Error is not auto-fixable: {strategy.description}")
                    raise
                
                # Record the attempt
                self.error_agent.record_attempt(job.id, strategy)
                
                # Apply the fix
                logger.info(f"🔧 Auto-recovery: {strategy.description}")
                _, plan = self.error_agent.apply_fix(strategy, {}, plan)
                
                # If switched to SDXL, rebuild plan arch
                if strategy.switch_to_sdxl:
                    plan.model_arch = "sdxl"
                    model_defaults = self.registry.get_model_defaults(plan.checkpoint)
                    if model_defaults:
                        plan.sampler = model_defaults.get("sampler", plan.sampler)
                        plan.scheduler = model_defaults.get("scheduler", plan.scheduler)
                
                # Notify user about the recovery
                interaction = self._job_interactions.get(job.id)
                if interaction:
                    try:
                        await interaction.channel.send(
                            f"⚠️ **Auto-Recovery:** {strategy.description}\n"
                            f"Retrying... (attempt {attempt + 2}/{max_retries + 1})"
                        )
                    except Exception:
                        pass
                
                # Update status
                if status_callback:
                    await status_callback(JobStatus.BUILDING)
        
        raise RuntimeError("All recovery attempts exhausted")

    async def _on_job_status_change(self, job: Job):
        """Called whenever a job's status changes — updates Discord."""
        await self._notify_discord(job)

    async def _notify_discord(self, job: Job):
        """Send status updates and results to Discord."""
        interaction = self._job_interactions.get(job.id)
        channel = self._job_channels.get(job.id)
        
        if not interaction and not channel:
            return
        
        dest = interaction.channel if interaction else channel
        
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
                # Include recovery info in error message
                error_msg = job.error
                recovery_info = ""
                retry_history = self.error_agent._retry_history.get(job.id, [])
                if retry_history:
                    recovery_info = "\n\n**Recovery attempts:**\n"
                    for i, attempt in enumerate(retry_history):
                        recovery_info += f"{i+1}. {attempt['description']}\n"
                
                embed = create_error_embed(error_msg + recovery_info, job)
                
                # Add LLM provider status if relevant
                provider_status = self.llm.get_status_summary()
                embed.add_field(
                    name="🤖 LLM Status",
                    value=provider_status[:200],
                    inline=False
                )
                
                await dest.send(embed=embed)
                
                # Clean up
                self.error_agent.clear_history(job.id)
                if job.id in self._job_interactions:
                    del self._job_interactions[job.id]
                if job.id in self._job_channels:
                    del self._job_channels[job.id]
            
            else:
                # Progress updates
                logger.debug(f"Job {job.id}: {job.status.value}")
        
        except Exception as e:
            logger.error(f"Failed to notify Discord for job {job.id}: {e}")
