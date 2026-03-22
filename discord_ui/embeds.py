"""
Discord Embeds — rich visual messages for the bot.
"""
import discord
from typing import Dict, Any, Optional


# ═══════════════════════════════════════════════════════════════
# Generation Flow Embeds
# ═══════════════════════════════════════════════════════════════

def create_generating_embed(prompt: str, workflow_name: str = "") -> discord.Embed:
    """Status embed while generating."""
    embed = discord.Embed(
        title="🎨 Generating...",
        description=f"**Prompt:** {prompt[:200]}",
        color=0xf59e0b,
    )
    if workflow_name:
        embed.add_field(name="Workflow", value=workflow_name, inline=True)
    embed.set_footer(text="This may take a moment...")
    return embed


def create_result_embed(prompt: str, workflow: str = "", seed: int = 0) -> discord.Embed:
    """Embed for a successfully generated image."""
    embed = discord.Embed(
        title="✨ Image Generated",
        color=0x22c55e,
    )
    embed.add_field(name="Prompt", value=prompt[:1024], inline=False)
    if workflow:
        embed.add_field(name="Workflow", value=workflow, inline=True)
    if seed:
        embed.add_field(name="Seed", value=str(seed), inline=True)
    return embed


def create_error_embed(error_message: str) -> discord.Embed:
    """Error embed."""
    embed = discord.Embed(
        title="❌ Generation Failed",
        description=error_message[:2000],
        color=0xef4444,
    )
    embed.set_footer(text="Try a different prompt or use /help")
    return embed


# ═══════════════════════════════════════════════════════════════
# Chat & Agentic Embeds
# ═══════════════════════════════════════════════════════════════

def create_chat_embed(message: str, user_id: str = "") -> discord.Embed:
    """Embed for AI chat responses."""
    embed = discord.Embed(
        description=message[:4096],
        color=0x8b5cf6,
    )
    embed.set_author(name="AI Director", icon_url=None)
    return embed


def create_plan_review_embed(plan, profile=None) -> discord.Embed:
    """Embed showing the generation plan for user review before generating."""
    embed = discord.Embed(
        title="📋 Generation Plan — Review & Confirm",
        description="Here's what I'm about to generate. Confirm or edit before I start!",
        color=0x6366f1,
    )

    embed.add_field(name="✍️ Enhanced Prompt", value=plan.enhanced_prompt[:1024], inline=False)
    embed.add_field(name="🔧 Workflow", value=plan.workflow_template or "auto-select", inline=True)
    embed.add_field(name="📐 Dimensions", value=f"{plan.width}×{plan.height}", inline=True)

    if plan.reasoning:
        embed.add_field(name="💭 Reasoning", value=plan.reasoning[:256], inline=False)

    if profile:
        caps = ", ".join(sorted(profile.capabilities)) if profile.capabilities else "general"
        embed.add_field(name="⚡ Capabilities", value=caps, inline=True)

        if profile.requires_image:
            imgs = ", ".join(img.purpose for img in profile.image_inputs)
            embed.add_field(name="🖼️ Images Needed", value=f"{profile.min_images} ({imgs})", inline=True)

    if plan.images:
        embed.add_field(name="📎 Attached Images", value=f"{len(plan.images)} uploaded", inline=True)
    elif plan.needs_image:
        embed.add_field(name="⚠️ Images Required", value="This workflow needs images — attach them!", inline=False)

    embed.set_footer(text="Click ✅ Generate to proceed, or ✏️ Edit to modify")
    return embed


# ═══════════════════════════════════════════════════════════════
# Workflow & Info Embeds
# ═══════════════════════════════════════════════════════════════

def create_workflow_list_embed(
    workflow_descs: Dict[str, str],
    profiles: Optional[Dict[str, Any]] = None,
) -> discord.Embed:
    """List available workflows with enriched metadata."""
    embed = discord.Embed(
        title="🔧 Available Workflows",
        description=f"{len(workflow_descs)} workflow(s) loaded",
        color=0x3b82f6,
    )

    for name, desc in workflow_descs.items():
        value_parts = [desc[:150]]

        if profiles and name in profiles:
            p = profiles[name]
            caps = p.get("capabilities", [])
            if caps:
                value_parts.append(f"**Caps:** {', '.join(caps[:5])}")
            if p.get("requires_image"):
                value_parts.append(f"**Images:** {p.get('min_images', '?')} required")
            arch = p.get("architecture", "unknown")
            if arch != "unknown":
                value_parts.append(f"**Arch:** {arch}")

        embed.add_field(name=f"📄 {name}", value="\n".join(value_parts), inline=False)

    embed.set_footer(text="Use /generate workflow:<name> to use a specific workflow")
    return embed


def create_help_embed() -> discord.Embed:
    """Help embed with usage instructions."""
    embed = discord.Embed(
        title="🎨 AI Director — Help",
        description="I'm your AI creative assistant! Here's how to use me:",
        color=0x8b5cf6,
    )

    embed.add_field(
        name="💬 Chat with me",
        value="@mention me to chat! I can discuss art, help refine ideas, and generate images from conversation.",
        inline=False,
    )
    embed.add_field(
        name="🖼️ Generate Images",
        value=(
            "`/generate <prompt>` — Generate an image from a text prompt\n"
            "`/generate <prompt> image:<file>` — Generate with a reference image\n"
            "`/generate <prompt> workflow:<name>` — Use a specific workflow"
        ),
        inline=False,
    )
    embed.add_field(
        name="📋 Other Commands",
        value=(
            "`/workflows` — List available generation workflows\n"
            "`/queue` — Check generation queue status\n"
            "`/status` — Check bot and system status\n"
            "`/upload_workflow` — Upload a new workflow (admin only)"
        ),
        inline=False,
    )
    embed.add_field(
        name="🖼️ Image Buttons",
        value="After generation, use:\n🔄 Retry | 🎲 Vary | ↔️ Wide | ↕️ Tall | 🗑️ Delete",
        inline=False,
    )

    embed.set_footer(text="Tip: Attach images to chat messages for image editing/style transfer!")
    return embed


# ═══════════════════════════════════════════════════════════════
# System Embeds
# ═══════════════════════════════════════════════════════════════

def create_queue_embed(position: int = 0, total: int = 0) -> discord.Embed:
    """Queue status embed."""
    if total == 0:
        embed = discord.Embed(
            title="📊 Queue Status",
            description="No jobs in queue — ready to generate!",
            color=0x22c55e,
        )
    else:
        embed = discord.Embed(
            title="📊 Queue Status",
            description=f"**{total}** job(s) in queue",
            color=0xf59e0b,
        )
        if position > 0:
            embed.add_field(name="Your Position", value=f"#{position}", inline=True)
    return embed


def create_status_embed(llm_name: str, workflow_count: int, config: dict = None) -> discord.Embed:
    """System status embed."""
    embed = discord.Embed(
        title="🤖 System Status",
        color=0x3b82f6,
    )

    embed.add_field(name="LLM Provider", value=llm_name, inline=True)
    embed.add_field(name="Workflows Loaded", value=str(workflow_count), inline=True)

    comfyui_url = config.get("comfyui", {}).get("url", "unknown") if config else "unknown"
    embed.add_field(name="ComfyUI Server", value=comfyui_url, inline=True)

    return embed
