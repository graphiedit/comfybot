"""
Discord Interactive Buttons — post-generation action buttons.

Retry, Upscale, Vary (subtle/strong), and Style change.
Each button triggers a new job with modified parameters.
"""
import discord
from discord.ui import View, Button, button

import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class GenerationButtons(View):
    """
    Action buttons shown after image generation.
    
    Buttons:
    - 🔄 Retry — same settings, new seed
    - ⬆️ Upscale — 2x resolution
    - 🎨 Vary (Subtle) — small variations
    - 🎭 Vary (Strong) — bigger variations
    """

    def __init__(
        self,
        job_id: str,
        on_action: Callable,
        timeout: float = 600,  # 10 minutes
    ):
        super().__init__(timeout=timeout)
        self.job_id = job_id
        self.on_action = on_action  # async callback(interaction, action, job_id)

    @button(label="Retry", emoji="🔄", style=discord.ButtonStyle.secondary, custom_id="retry")
    async def retry_button(self, interaction: discord.Interaction, btn: Button):
        await self._handle_action(interaction, "retry")

    @button(label="Upscale 2x", emoji="⬆️", style=discord.ButtonStyle.primary, custom_id="upscale")
    async def upscale_button(self, interaction: discord.Interaction, btn: Button):
        await self._handle_action(interaction, "upscale")

    @button(label="Vary (Subtle)", emoji="🎨", style=discord.ButtonStyle.secondary, custom_id="vary_subtle")
    async def vary_subtle_button(self, interaction: discord.Interaction, btn: Button):
        await self._handle_action(interaction, "vary_subtle")

    @button(label="Vary (Strong)", emoji="🎭", style=discord.ButtonStyle.secondary, custom_id="vary_strong")
    async def vary_strong_button(self, interaction: discord.Interaction, btn: Button):
        await self._handle_action(interaction, "vary_strong")

    @button(label="Info", emoji="ℹ️", style=discord.ButtonStyle.secondary, custom_id="info")
    async def info_button(self, interaction: discord.Interaction, btn: Button):
        await self._handle_action(interaction, "info")

    async def _handle_action(self, interaction: discord.Interaction, action: str):
        """Route button clicks to the action callback."""
        try:
            await self.on_action(interaction, action, self.job_id)
        except Exception as e:
            logger.error(f"Button action '{action}' failed: {e}", exc_info=True)
            try:
                await interaction.response.send_message(
                    f"❌ Action failed: {str(e)[:200]}",
                    ephemeral=True,
                )
            except discord.errors.InteractionResponded:
                pass

    async def on_timeout(self):
        """Disable buttons after timeout."""
        for item in self.children:
            if isinstance(item, Button):
                item.disabled = True


class ModelSelectView(View):
    """
    Dynamic model selection dropdown — lets users pick a checkpoint.
    """

    def __init__(
        self,
        checkpoints: list,
        on_select: Callable,
        timeout: float = 120,
    ):
        super().__init__(timeout=timeout)
        self.on_select = on_select
        
        # Build select menu
        options = []
        for ckpt in checkpoints[:25]:  # Discord limit: 25 options
            name = ckpt if isinstance(ckpt, str) else ckpt.get("filename", "")
            label = name.replace(".safetensors", "")[:100]
            
            # Add style hint if available
            desc = ""
            if isinstance(ckpt, dict) and ckpt.get("styles"):
                desc = ", ".join(ckpt["styles"])[:100]
            
            options.append(
                discord.SelectOption(label=label, value=name, description=desc)
            )
        
        if options:
            select = discord.ui.Select(
                placeholder="Choose a model...",
                options=options,
                custom_id="model_select",
            )
            select.callback = self._on_model_selected
            self.add_item(select)

    async def _on_model_selected(self, interaction: discord.Interaction):
        selected = interaction.data["values"][0]
        await self.on_select(interaction, selected)


class StyleSelectView(View):
    """Quick style selection buttons."""

    STYLES = [
        ("📸 Realistic", "realistic"),
        ("🎌 Anime", "anime"),
        ("🎬 Cinematic", "cinematic"),
        ("🗡️ Fantasy", "fantasy"),
        ("🤖 Sci-Fi", "scifi"),
        ("🎨 Artistic", "artistic"),
    ]

    def __init__(self, on_select: Callable, timeout: float = 120):
        super().__init__(timeout=timeout)
        self.on_select = on_select
        
        for label, value in self.STYLES:
            btn = Button(label=label, style=discord.ButtonStyle.secondary, custom_id=f"style_{value}")
            btn.callback = self._make_callback(value)
            self.add_item(btn)

    def _make_callback(self, style: str):
        async def callback(interaction: discord.Interaction):
            await self.on_select(interaction, style)
        return callback
