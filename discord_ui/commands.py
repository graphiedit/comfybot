"""
Discord Slash Commands — the user-facing interface.

Commands:
  /generate    — generate an image from a prompt (+ optional reference image)
  /upscale     — upscale a previously generated image
  /vary        — create variations of a previous generation
  /edit        — edit an image with a new prompt (img2img)
  /chat        — chat with the AI Director about art
  /models      — list available models
  /queue       — check queue status
  /settings    — view/change generation defaults
  
  # Director Features
  /dream       — Auto/Creative mode (minimal input)
  /refine      — Give feedback on the last generation
  /style       — Browse and apply style presets
  /history     — View recent generations
  /preferences — View/manage your user preferences
"""
import io
import dataclasses
import logging

import discord
from discord import app_commands
from typing import Optional, List, Dict, Any

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
        workflow="Select a custom workflow template",
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
        workflow: Optional[str] = None,
    ):
        await interaction.response.defer()
        
        # Build user overrides
        overrides: Dict[str, Any] = {}
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
        if workflow:
            overrides["workflow_template"] = workflow
        
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

    @generate_cmd.autocomplete("workflow")
    async def workflow_autocomplete(
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        templates = bot.engine.workflow_manager.templates
        return [
            app_commands.Choice(name=name, value=name)
            for name in templates.keys() if current.lower() in name.lower()
        ][:25]

    @bot.tree.command(name="workflows", description="📜 List available custom workflow templates")
    async def workflows_cmd(interaction: discord.Interaction):
        templates = bot.engine.workflow_manager.templates
        if not templates:
            await interaction.response.send_message("No custom workflow templates found in `data/workflows/`.", ephemeral=True)
            return
            
        from discord_ui.embeds import create_workflows_list_embed
        embed = create_workflows_list_embed(templates)
        await interaction.response.send_message(embed=embed, ephemeral=True)

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

    # --- Director Features ---

    @bot.tree.command(name="dream", description="✨ Auto Mode: AI Director invents a prompt for you")
    @app_commands.describe(
        theme="Optional theme or basic idea (e.g., 'cyberpunk city')",
    )
    async def dream_cmd(
        interaction: discord.Interaction,
        theme: Optional[str] = None,
    ):
        await interaction.response.defer()
        prompt = theme or "Surprise me with a beautiful, creative masterpiece"
        await bot.engine.submit_generation(
            interaction=interaction,
            prompt=prompt,
            user_overrides={"action": "dream"},
        )

    @bot.tree.command(name="refine", description="🖌️ Refine your last generation with feedback")
    @app_commands.describe(
        feedback="What to change (e.g., 'make it darker', 'fix the eyes')",
    )
    async def refine_cmd(
        interaction: discord.Interaction,
        feedback: str,
    ):
        # We need to find the user's last generated image
        last_job_meta = getattr(bot.engine, "_user_last_job", {}).get(str(interaction.user.id))
        
        if not last_job_meta or "result_images" not in last_job_meta:
            await interaction.response.send_message("❌ I couldn't find your last generation. Use `/generate` or `/dream` first.", ephemeral=True)
            return

        await interaction.response.defer()
        image_bytes = last_job_meta["result_images"][-1]
        
        # We generate a refined prompt based on feedback
        # A full implementation would use the LLM to understand the feedback
        # For now, we append it to the prompt if we know it, or just use it roughly
        original_prompt = last_job_meta.get("prompt", "")
        new_prompt = f"{original_prompt}, {feedback}" if original_prompt else feedback
        
        await bot.engine.submit_generation(
            interaction=interaction,
            prompt=new_prompt,
            image_bytes=image_bytes,
            user_overrides={"action": "refine_feedback", "denoise": 0.45},
        )

    @bot.tree.command(name="style", description="🎭 Browse available style presets")
    async def style_cmd(interaction: discord.Interaction):
        from discord_ui.buttons import StyleSelectView
        
        embed = discord.Embed(
            title="🎨 Style Presets",
            description="Select a style below to see what it looks like or use it in your next generation. (You can also type the style directly like `/generate prompt: cat style: cinematic`).",
            color=0x7C3AED
        )
        
        async def on_style_select(interaction: discord.Interaction, style: str):
            await interaction.response.send_message(f"Selected style: **{style}**. Try `/generate style:{style}`", ephemeral=True)
            
        view = StyleSelectView(on_select=on_style_select)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
        
    @bot.tree.command(name="preferences", description="👤 View your AI Director preferences")
    async def prefs_cmd(interaction: discord.Interaction):
        user_id = str(interaction.user.id)
        prefs = bot.engine.user_memory.get_profile(user_id)
        
        embed = discord.Embed(
            title=f"👤 Preferences for {interaction.user.display_name}",
            color=0x7C3AED
        )
        embed.add_field(name="Generations", value=str(prefs.generation_count))
        
        top_styles = [f"{s} ({c})" for s, c in prefs.get_top_styles(3)]
        embed.add_field(name="Top Styles", value=", ".join(top_styles) or "None yet")
        
        top_models = [f"{m.replace('.safetensors', '')} ({c})" for m, c in prefs.get_top_models(3)]
        embed.add_field(name="Top Models", value=", ".join(top_models) or "None yet")
        
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
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
