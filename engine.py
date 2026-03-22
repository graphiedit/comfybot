"""
AI Director Engine — central orchestrator for the ComfyBot.

Connects: LLM (with failover) → Workflow Manager (with analyzer) → 
          ComfyUI Client (with image upload) → Error Recovery → Pipeline.

This is the beating heart of the agentic AI system.
"""
import asyncio
import logging
import random
import tempfile
import os
from typing import Optional, List, Dict, Any

from llm.base import GenerationPlan, ChatResponse, create_llm_provider
from core.comfyui_client import ComfyUIClient
from core.workflow_manager import WorkflowManager
from core.error_recovery import ErrorDiagnosticAgent
from core.user_memory import UserMemory

logger = logging.getLogger(__name__)


class AIDirectorEngine:
    """
    The AI Director — orchestrates the full agentic generation pipeline.
    
    Flow:
    1. User sends message/command
    2. LLM analyzes intent → GenerationPlan
    3. Upload any user images to ComfyUI
    4. Build workflow from plan
    5. Submit to ComfyUI → wait for result
    6. On error → ErrorRecovery → auto-fix → retry
    7. Return images to user
    """

    def __init__(self, config: dict):
        self.config = config

        # Core components
        self.llm = create_llm_provider(config)
        self.comfyui = ComfyUIClient(config.get("comfyui", {}))
        self.workflow_manager = WorkflowManager()

        self.error_agent = ErrorDiagnosticAgent()
        self.user_memory = UserMemory(config)

        # Pipeline settings
        pipeline_cfg = config.get("pipeline", {})
        self.max_retries = pipeline_cfg.get("max_retries", 3)

        # Conversation memory: channel_id -> [messages]
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        # Active generations: channel_id -> plan (for vary/retry)
        self.active_generations: Dict[str, GenerationPlan] = {}

        # Check if LLM needs async initialization (ProviderManager)
        self._llm_initialized = False

    async def _ensure_llm_initialized(self):
        """Initialize ProviderManager if we haven't already."""
        if self._llm_initialized:
            return
        # ProviderManager has an async initialize() method
        if hasattr(self.llm, 'initialize'):
            await self.llm.initialize()
        self._llm_initialized = True

    # ═══════════════════════════════════════════════════════════════
    # IMAGE GENERATION — The main pipeline
    # ═══════════════════════════════════════════════════════════════

    async def generate(
        self,
        prompt: str,
        user_id: str,
        channel,
        workflow_override: str = "",
        images: Optional[List[bytes]] = None,
        image_filenames: Optional[List[str]] = None,
    ) -> List[bytes]:
        """
        Generate images from a text prompt.
        
        Args:
            prompt: User's text prompt
            user_id: Discord user ID for preferences
            channel: Discord channel for status messages
            workflow_override: Force a specific workflow template
            images: Raw image bytes from Discord attachments
            image_filenames: Already-named image files (for retry/vary)
        
        Returns:
            List of image bytes, or empty list on failure
        """
        await self._ensure_llm_initialized()

        # Upload images to ComfyUI if provided
        uploaded_images = list(image_filenames or [])
        if images:
            for i, img_data in enumerate(images):
                try:
                    filename = f"discord_upload_{user_id}_{i}.png"
                    uploaded_name = await self.comfyui.upload_image_bytes(img_data, filename)
                    uploaded_images.append(uploaded_name)
                except Exception as e:
                    logger.error(f"Failed to upload image {i}: {e}")

        has_images = len(uploaded_images) > 0

        # Step 1: AI Intent Analysis
        user_prefs = self.user_memory.get_preferences_for_llm(user_id)
        workflow_summaries = self.workflow_manager.get_workflow_summaries_for_llm()

        try:
            plan = await self.llm.analyze_intent(
                prompt=f"{prompt}\n\n[User preferences: {user_prefs}]\n[Has attached images: {has_images}, count: {len(uploaded_images)}]",
                workflows=self.workflow_manager.get_workflow_list(),
            )
        except Exception as e:
            logger.error(f"LLM intent analysis failed: {e}")
            # Fallback: use first text-to-image workflow
            plan = GenerationPlan(
                workflow_template=next(iter(self.workflow_manager.templates), ""),
                enhanced_prompt=prompt,
                reasoning=f"Fallback (LLM error): {e}",
            )

        # Apply workflow override if specified
        if workflow_override:
            plan.workflow_template = workflow_override

        # Inject uploaded images into the plan
        plan.images = uploaded_images

        # Check if this workflow needs images but none were provided
        profile = self.workflow_manager.get_profile(plan.workflow_template)
        if profile and profile.requires_image and not uploaded_images:
            plan.needs_image = True
            logger.warning(f"Workflow '{plan.workflow_template}' requires images but none provided")

        # Store for retry/vary
        channel_id = str(getattr(channel, 'id', channel))
        self.active_generations[channel_id] = plan

        # Step 2: Build & Submit with Error Recovery
        result_images = await self._execute_with_recovery(plan, channel)

        # Step 3: Record to user memory
        if result_images:
            try:
                self.user_memory.record_generation(user_id, plan)
            except Exception as e:
                logger.debug(f"Failed to record generation: {e}")

        return result_images

    async def _execute_with_recovery(self, plan: GenerationPlan, channel) -> List[bytes]:
        """Execute generation with automatic error recovery and retry."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Build workflow
                workflow, success = self.workflow_manager.build_workflow(plan)
                if not success or not workflow:
                    logger.error("Failed to build workflow")
                    return []

                # Submit to ComfyUI
                prompt_id = await self.comfyui.queue_prompt(workflow)
                if not prompt_id:
                    logger.error("Failed to queue prompt")
                    return []

                # Wait for results
                images = await self.comfyui.wait_for_result(prompt_id)
                if images:
                    return images
                else:
                    logger.warning(f"No images returned (attempt {attempt + 1})")

            except Exception as e:
                last_error = e
                error_msg = str(e)
                logger.warning(f"Generation attempt {attempt + 1} failed: {error_msg}")

                if attempt < self.max_retries - 1:
                    # Try error recovery
                    try:
                        strategy = self.error_agent.diagnose(error_msg, workflow, plan)
                        if strategy.fixes:
                            logger.info(f"Error recovery: {strategy.error_type}, applying {len(strategy.fixes)} fixes")

                            # Apply fixes to the workflow
                            workflow = self.error_agent.apply_fixes(workflow, strategy)

                            # Apply fixes that affect the plan (e.g., reduce resolution)
                            if strategy.reduce_resolution:
                                plan.width = int(plan.width * 0.75)
                                plan.height = int(plan.height * 0.75)
                                logger.info(f"Reduced resolution to {plan.width}x{plan.height}")

                            # Retry with fixed workflow
                            try:
                                prompt_id = await self.comfyui.queue_prompt(workflow)
                                if prompt_id:
                                    images = await self.comfyui.wait_for_result(prompt_id)
                                    if images:
                                        return images
                            except Exception as retry_error:
                                logger.warning(f"Retry after fix failed: {retry_error}")
                                continue
                        else:
                            logger.info("No recovery strategy available")
                    except Exception as recovery_error:
                        logger.error(f"Error recovery itself failed: {recovery_error}")
                        continue

        logger.error(f"All {self.max_retries} generation attempts failed. Last error: {last_error}")
        return []

    # ═══════════════════════════════════════════════════════════════
    # CHAT — Conversational AI with generation triggers
    # ═══════════════════════════════════════════════════════════════

    async def chat(
        self,
        message: str,
        user_id: str,
        channel,
        images: Optional[List[bytes]] = None,
    ) -> ChatResponse:
        """
        Handle a conversational message. May trigger image generation.
        
        Args:
            message: The user's message text
            user_id: Discord user ID
            channel: Discord channel object
            images: Optional image attachments from the message
        """
        await self._ensure_llm_initialized()

        channel_id = str(getattr(channel, 'id', channel))

        # Get/create conversation history
        if channel_id not in self.conversations:
            self.conversations[channel_id] = []
        history = self.conversations[channel_id]

        # Add user message to history
        history.append({"role": "user", "content": message})

        # Build system context with user preferences and workflow info
        user_prefs = self.user_memory.get_preferences_for_llm(user_id)
        has_images = bool(images) and len(images) > 0
        workflow_summaries = self.workflow_manager.get_workflow_summaries_for_llm()

        system_context = (
            f"User preferences: {user_prefs}\n"
            f"User has attached {len(images) if images else 0} image(s) to this message.\n"
            f"Available workflows:\n{workflow_summaries}"
        )

        try:
            response = await self.llm.chat(
                message=message,
                conversation_history=history,
                workflows=self.workflow_manager.get_workflow_list(),
            )
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            response = ChatResponse(
                message="I'm having trouble thinking right now. Try again, or use /generate directly! 🎨"
            )

        # Add AI response to history
        history.append({"role": "assistant", "content": response.message})

        # Keep history manageable
        if len(history) > 20:
            history[:] = history[-20:]

        # If the AI wants to generate, trigger it
        if response.should_generate:
            gen_prompt = response.generation_prompt
            workflow_hint = response.workflow_hint or ""

            # Fire and forget — generate in background
            asyncio.create_task(
                self._generate_from_chat(
                    prompt=gen_prompt,
                    user_id=user_id,
                    channel=channel,
                    workflow_hint=workflow_hint,
                    images=images,
                )
            )

        return response

    async def _generate_from_chat(
        self,
        prompt: str,
        user_id: str,
        channel,
        workflow_hint: str = "",
        images: Optional[List[bytes]] = None,
    ):
        """Generate images triggered from a chat conversation."""
        try:
            result = await self.generate(
                prompt=prompt,
                user_id=user_id,
                channel=channel,
                workflow_override=workflow_hint,
                images=images,
            )

            if result:
                # Send images to channel
                import discord
                import io
                for i, img_data in enumerate(result):
                    file = discord.File(io.BytesIO(img_data), filename=f"generated_{i}.png")
                    from discord_ui.embeds import create_result_embed
                    from discord_ui.buttons import ImageActionView

                    embed = create_result_embed(
                        prompt=prompt,
                        workflow="chat generation",
                        seed=random.randint(0, 2**31),
                    )
                    view = ImageActionView(engine=self, prompt=prompt, user_id=user_id)
                    await channel.send(file=file, embed=embed, view=view)
            else:
                await channel.send("❌ Generation failed — I couldn't produce the image. Try adjusting your prompt or use /generate directly.")

        except Exception as e:
            logger.error(f"Chat-triggered generation failed: {e}")
            try:
                await channel.send(f"❌ Oops, something went wrong: {str(e)[:200]}")
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════
    # ACTION HANDLERS — Retry, Vary, Upscale, etc.
    # ═══════════════════════════════════════════════════════════════

    async def retry_generation(self, channel, user_id: str) -> List[bytes]:
        """Retry the last generation with a new seed."""
        channel_id = str(getattr(channel, 'id', channel))
        plan = self.active_generations.get(channel_id)
        if not plan:
            return []

        plan.seed = random.randint(0, 2**31)
        return await self._execute_with_recovery(plan, channel)

    async def vary_generation(self, channel, user_id: str) -> List[bytes]:
        """Generate a variation with slightly modified prompt."""
        await self._ensure_llm_initialized()

        channel_id = str(getattr(channel, 'id', channel))
        plan = self.active_generations.get(channel_id)
        if not plan:
            return []

        try:
            new_prompt = await self.llm.enhance_prompt(
                plan.enhanced_prompt,
                style_hints="Create a variation — keep the same subject but change style, lighting, or composition slightly.",
            )
            plan.enhanced_prompt = new_prompt
        except Exception as e:
            logger.warning(f"Variation prompt enhancement failed: {e}")

        plan.seed = random.randint(0, 2**31)
        return await self._execute_with_recovery(plan, channel)

    async def change_aspect(self, channel, user_id: str, direction: str = "wide") -> List[bytes]:
        """Regenerate with different aspect ratio."""
        channel_id = str(getattr(channel, 'id', channel))
        plan = self.active_generations.get(channel_id)
        if not plan:
            return []

        if direction == "wide":
            plan.width = 1344
            plan.height = 768
        elif direction == "tall":
            plan.width = 768
            plan.height = 1344

        plan.seed = random.randint(0, 2**31)
        return await self._execute_with_recovery(plan, channel)

    # ═══════════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════════

    def get_workflows(self) -> Dict[str, str]:
        """Get all available workflows."""
        return self.workflow_manager.get_workflow_list()

    def get_workflow_profiles(self) -> Dict[str, Any]:
        """Get enriched workflow profiles."""
        return self.workflow_manager.get_workflow_list_rich()
