"""
Discord Slash Commands — the user-facing interface.

Commands:
  /generate  — generate an image from a prompt (+ optional reference image)
  /upscale   — upscale a previously generated image
  /vary      — create variations of a previous generation
  /edit      — edit an image with a new prompt (img2img)
  /chat      — chat with the AI Director about art
  /models    — list available models
  /queue     — check queue status
  /settings  — view/change generation defaults
"""
import io
import dataclasses
import logging

import discord
from discord import app_commands
from typing import Optional

logger = logging.getLogger(__name__)


def setup_commands(bot):
    """Register all slash commands on the bot."""

    @bot.tree.command(name="generate", description="🎨 Generate an image from a text prompt")
    @app_commands.describe(
        prompt="Describe the image you want to create",
        image="Optional reference image for style/composition",
        model="Override model selection (leave empty for auto)",
        style="Force a style: realistic, anime, cinematic, fantasy, artistic",
        steps="Number of sampling steps (default: auto)",
        seed="Random seed (-1 for random)",
        width="Image width (default: 1024)",
        height="Image height (default: 1024)",
    )
    async def generate_cmd(
        interaction: discord.Interaction,
        prompt: str,
        image: Optional[discord.Attachment] = None,
        model: Optional[str] = None,
        style: Optional[str] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        await interaction.response.defer()
        
        # Build user overrides
        overrides = {}
        if model:
            overrides["checkpoint"] = model
        if style:
            overrides["style_category"] = style
        if steps:
            overrides["steps"] = steps
        if seed is not None:
            overrides["seed"] = seed
        if width:
            overrides["width"] = width
        if height:
            overrides["height"] = height
        
        # Handle reference image
        image_bytes = None
        if image:
            image_bytes = await image.read()
        
        # Submit to the engine
        await bot.engine.submit_generation(
            interaction=interaction,
            prompt=prompt,
            image_bytes=image_bytes,
            user_overrides=overrides,
        )

    @bot.tree.command(name="upscale", description="⬆️ Upscale a previously generated image")
    @app_commands.describe(
        image="Image to upscale (attach an image)",
        scale="Upscale factor (default: 2)",
    )
    async def upscale_cmd(
        interaction: discord.Interaction,
        image: discord.Attachment,
        scale: Optional[int] = 2,
    ):
        await interaction.response.defer()
        
        image_bytes = await image.read()
        await bot.engine.submit_upscale(
            interaction=interaction,
            image_bytes=image_bytes,
            scale=scale,
        )

    @bot.tree.command(name="vary", description="🎭 Create variations of a prompt or image")
    @app_commands.describe(
        prompt="The prompt to create variations of",
        image="Optional image to vary",
        strength="Variation strength: subtle (0.3) or strong (0.7)",
    )
    async def vary_cmd(
        interaction: discord.Interaction,
        prompt: str,
        image: Optional[discord.Attachment] = None,
        strength: Optional[str] = "subtle",
    ):
        await interaction.response.defer()
        
        image_bytes = None
        if image:
            image_bytes = await image.read()
        
        denoise = 0.3 if strength == "subtle" else 0.7
        
        await bot.engine.submit_generation(
            interaction=interaction,
            prompt=prompt,
            image_bytes=image_bytes,
            user_overrides={"denoise": denoise, "action": "vary"},
        )

    @bot.tree.command(name="edit", description="✏️ Edit an image with a new prompt")
    @app_commands.describe(
        prompt="Describe the changes you want",
        image="The image to edit",
        strength="How much to change (0.0-1.0, default 0.6)",
    )
    async def edit_cmd(
        interaction: discord.Interaction,
        prompt: str,
        image: discord.Attachment,
        strength: Optional[float] = 0.6,
    ):
        await interaction.response.defer()
        
        image_bytes = await image.read()
        await bot.engine.submit_generation(
            interaction=interaction,
            prompt=prompt,
            image_bytes=image_bytes,
            user_overrides={"denoise": strength, "action": "edit"},
        )

    @bot.tree.command(name="chat", description="💬 Chat with the AI Director about art and generation")
    @app_commands.describe(message="Your message to the AI Director")
    async def chat_cmd(
        interaction: discord.Interaction,
        message: str,
    ):
        await interaction.response.defer()
        
        from discord_ui.embeds import create_chat_embed
        
        try:
            response = await bot.engine.chat(
                user_id=str(interaction.user.id),
                message=message,
            )
            embed = create_chat_embed(response)
            await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Chat failed: {e}", exc_info=True)
            await interaction.followup.send(f"❌ Chat error: {str(e)[:200]}")

    @bot.tree.command(name="models", description="📋 List available models and LoRAs")
    async def models_cmd(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        available = bot.engine.registry.get_available_models_for_llm()
        
        lines = ["**📦 Available Models**\n"]
        
        if available.get("checkpoints"):
            lines.append("__Checkpoints:__")
            for ckpt in available["checkpoints"]:
                name = ckpt["filename"] if isinstance(ckpt, dict) else ckpt
                styles = ""
                if isinstance(ckpt, dict) and ckpt.get("styles"):
                    styles = f" (`{', '.join(ckpt['styles'])}`)"
                arch = ""
                if isinstance(ckpt, dict) and ckpt.get("arch"):
                    arch = f" [{ckpt['arch'].upper()}]"
                lines.append(f"• `{name}`{styles}{arch}")
        
        if available.get("diffusion_models"):
            lines.append("\n__Diffusion Models (Flux/Hunyuan):__")
            for dm in available["diffusion_models"]:
                name = dm["filename"] if isinstance(dm, dict) else dm
                arch = ""
                if isinstance(dm, dict) and dm.get("arch"):
                    arch = f" [{dm['arch'].upper()}]"
                styles = ""
                if isinstance(dm, dict) and dm.get("styles"):
                    styles = f" (`{', '.join(dm['styles'])}`)"
                lines.append(f"• `{name}`{arch}{styles}")
        
        if available.get("loras"):
            lines.append("\n__LoRAs:__")
            for lora in available["loras"][:15]:
                name = lora["filename"] if isinstance(lora, dict) else lora
                kw = ""
                if isinstance(lora, dict) and lora.get("keywords"):
                    kw = f" — {', '.join(lora['keywords'][:3])}"
                lines.append(f"• `{name}`{kw}")
        
        if available.get("ipadapters"):
            lines.append("\n__IP-Adapters:__")
            for ipa in available["ipadapters"]:
                name = ipa["filename"] if isinstance(ipa, dict) else ipa
                lines.append(f"• `{name}`")
        
        if available.get("controlnets"):
            lines.append("\n__ControlNets:__")
            for cn in available["controlnets"]:
                name = cn["filename"] if isinstance(cn, dict) else cn
                lines.append(f"• `{name}`")
        
        text = "\n".join(lines)
        if len(text) > 2000:
            text = text[:1997] + "..."
        
        await interaction.followup.send(text, ephemeral=True)

    @bot.tree.command(name="queue", description="📊 Check the generation queue status")
    async def queue_cmd(interaction: discord.Interaction):
        from discord_ui.embeds import create_queue_embed
        
        queue_size = bot.engine.queue.get_queue_size()
        active_count = bot.engine.queue.get_active_count()
        
        embed = create_queue_embed(queue_size, active_count)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @bot.tree.command(name="settings", description="⚙️ View current default generation settings")
    async def settings_cmd(interaction: discord.Interaction):
        defaults = bot.engine.config.get("defaults", {})
        
        embed = discord.Embed(
            title="⚙️ Default Settings",
            color=0x7C3AED,
        )
        embed.add_field(name="Steps", value=str(defaults.get("steps", 30)), inline=True)
        embed.add_field(name="CFG", value=str(defaults.get("cfg", 7.0)), inline=True)
        embed.add_field(name="Sampler", value=defaults.get("sampler", "dpmpp_2m_sde"), inline=True)
        embed.add_field(name="Scheduler", value=defaults.get("scheduler", "karras"), inline=True)
        embed.add_field(name="Resolution", value=f"{defaults.get('width', 1024)}×{defaults.get('height', 1024)}", inline=True)
        embed.add_field(name="Model", value=defaults.get("ckpt", "Juggernaut-XL_v9"), inline=True)
        embed.add_field(
            name="LLM Provider",
            value=bot.engine.config.get("llm", {}).get("provider", "ollama"),
            inline=True,
        )
        
        await interaction.response.send_message(embed=embed, ephemeral=True)

    logger.info("Slash commands registered")
