"""
Discord Rich Embeds — beautiful, informative message formatting.

Creates progress indicators, generation results with metadata,
and error states with helpful messages.
"""
import discord
from typing import Optional

from core.queue_manager import Job, JobStatus
from core.quality_analyzer import QualityScore

# Color palette
COLOR_PRIMARY = 0x7C3AED     # Purple — main brand
COLOR_SUCCESS = 0x10B981     # Green — completed
COLOR_WARNING = 0xF59E0B     # Amber — warnings
COLOR_ERROR = 0xEF4444       # Red — errors
COLOR_INFO = 0x3B82F6        # Blue — info
COLOR_GENERATING = 0x8B5CF6  # Light purple — in progress


def create_progress_embed(job: Job, queue_position: int = 0) -> discord.Embed:
    """Create a progress embed showing generation status."""
    status_map = {
        JobStatus.QUEUED: ("⏳ In Queue", f"Position: #{queue_position}", COLOR_INFO),
        JobStatus.ANALYZING: ("🧠 Analyzing Prompt", "AI Director is understanding your request...", COLOR_GENERATING),
        JobStatus.BUILDING: ("🔧 Building Pipeline", "Constructing the optimal workflow...", COLOR_GENERATING),
        JobStatus.GENERATING: ("🎨 Generating Image", "ComfyUI is creating your masterpiece...", COLOR_GENERATING),
        JobStatus.UPSCALING: ("⬆️ Upscaling", "Enhancing resolution...", COLOR_GENERATING),
    }
    
    title, desc, color = status_map.get(
        job.status,
        ("⚙️ Processing", "Working on it...", COLOR_INFO),
    )
    
    embed = discord.Embed(title=title, description=desc, color=color)
    embed.set_footer(text=f"Job: {job.id}")
    
    if job.prompt:
        prompt_display = job.prompt[:100] + ("..." if len(job.prompt) > 100 else "")
        embed.add_field(name="Prompt", value=f"```{prompt_display}```", inline=False)
    
    return embed


def create_generation_embed(
    job: Job,
    plan_metadata: dict = None,
) -> discord.Embed:
    """Create a rich result embed showing what was generated and how."""
    elapsed = job.completed_at - job.started_at if job.completed_at else 0
    
    embed = discord.Embed(
        title="✨ Generation Complete",
        color=COLOR_SUCCESS,
    )
    
    if job.prompt:
        prompt_display = job.prompt[:200] + ("..." if len(job.prompt) > 200 else "")
        embed.add_field(name="📝 Prompt", value=prompt_display, inline=False)
    
    if plan_metadata:
        # Show what the AI chose
        details = []
        if plan_metadata.get("checkpoint"):
            model_name = plan_metadata["checkpoint"].replace(".safetensors", "")
            details.append(f"**Model**: {model_name}")
        if plan_metadata.get("style_category"):
            details.append(f"**Style**: {plan_metadata['style_category']}")
        if plan_metadata.get("loras"):
            lora_names = [l.get("name", "?").replace(".safetensors", "") for l in plan_metadata["loras"]]
            details.append(f"**LoRAs**: {', '.join(lora_names)}")
        if plan_metadata.get("use_controlnet"):
            details.append(f"**ControlNet**: {plan_metadata.get('controlnet_type', 'enabled')}")
        if plan_metadata.get("use_ipadapter"):
            details.append(f"**IP-Adapter**: weight {plan_metadata.get('ipadapter_weight', 0.6)}")
        
        if details:
            embed.add_field(name="🤖 AI Decisions", value="\n".join(details), inline=True)
        
        # Technical details
        tech = []
        if plan_metadata.get("steps"):
            tech.append(f"Steps: {plan_metadata['steps']}")
        if plan_metadata.get("cfg"):
            tech.append(f"CFG: {plan_metadata['cfg']}")
        if plan_metadata.get("sampler"):
            tech.append(f"Sampler: {plan_metadata['sampler']}")
        if plan_metadata.get("seed"):
            tech.append(f"Seed: {plan_metadata['seed']}")
        
        if tech:
            embed.add_field(name="⚙️ Settings", value="\n".join(tech), inline=True)
    
    if plan_metadata and plan_metadata.get("reasoning"):
        embed.add_field(
            name="💡 AI Reasoning",
            value=plan_metadata["reasoning"][:200],
            inline=False,
        )
    
    embed.set_footer(text=f"⏱️ {elapsed:.1f}s | Job: {job.id}")
    
    return embed


def create_error_embed(error: str, job: Optional[Job] = None) -> discord.Embed:
    """Create an error embed with helpful information."""
    embed = discord.Embed(
        title="❌ Generation Failed",
        description=error[:500],
        color=COLOR_ERROR,
    )
    
    # Add helpful troubleshooting
    if "connect" in error.lower() or "timed out" in error.lower():
        embed.add_field(
            name="💡 Tip",
            value="Make sure ComfyUI is running at the configured URL.",
            inline=False,
        )
    elif "checkpoint" in error.lower() or "model" in error.lower():
        embed.add_field(
            name="💡 Tip",
            value="The requested model may not be installed. Check your ComfyUI models folder.",
            inline=False,
        )
    elif "ollama" in error.lower():
        embed.add_field(
            name="💡 Tip",
            value="Make sure Ollama is running (`ollama serve`).",
            inline=False,
        )
    
    if job:
        embed.set_footer(text=f"Job: {job.id}")
    
    return embed


def create_queue_embed(queue_size: int, active_count: int) -> discord.Embed:
    """Create a queue status embed."""
    if active_count == 0 and queue_size == 0:
        desc = "🟢 No active jobs — ready to generate!"
    else:
        desc = f"🔵 {active_count} generating, {queue_size} waiting"
    
    embed = discord.Embed(
        title="📊 Queue Status",
        description=desc,
        color=COLOR_INFO,
    )
    return embed


def create_chat_embed(response: str) -> discord.Embed:
    """Create a chat response embed."""
    embed = discord.Embed(
        description=response,
        color=COLOR_PRIMARY,
    )
    embed.set_author(name="🎨 AI Director")
    return embed


def create_quality_report_embed(score: QualityScore) -> discord.Embed:
    """Display the results of the LLM quality analysis."""
    
    color = COLOR_SUCCESS if score.overall >= 7.0 else (COLOR_WARNING if score.overall >= 5.0 else COLOR_ERROR)
    
    embed = discord.Embed(
        title="🔍 Quality Analysis Report",
        description=f"**Overall Score:** {score.overall}/10",
        color=color
    )
    
    # Detailed scores
    details = []
    if score.faces is not None:
        details.append(f"**Faces:** {score.faces}/10")
    if score.hands is not None:
        details.append(f"**Hands:** {score.hands}/10")
    if score.composition is not None:
        details.append(f"**Composition:** {score.composition}/10")
    if score.sharpness is not None:
        details.append(f"**Sharpness:** {score.sharpness}/10")
    if score.artifacts is not None:
        details.append(f"**Artifacts:** {score.artifacts}/10 (higher is better)")
        
    embed.add_field(name="Detailed Scores", value="\n".join(details), inline=True)
    
    # Issues
    if score.issues:
        embed.add_field(name="Detected Issues", value="\n".join([f"• {i}" for i in score.issues]), inline=True)
        
    return embed
