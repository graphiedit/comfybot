"""
ComfyBot — Discord bot for AI image generation via ComfyUI.

Supports:
- @mention for conversational AI with auto-generation triggers
- /generate for direct image generation (with optional image attachments)
- /workflows to list available workflows with capabilities
- /queue to check generation queue status
- /status to check system health
- /upload_workflow for admins to add new workflows
- /help for usage information
"""
import asyncio
import io
import logging
import random
import traceback
import yaml
import discord
from discord.ext import commands

from engine import AIDirectorEngine
from discord_ui.commands import setup_commands
from discord_ui.embeds import (
    create_generating_embed, create_result_embed, create_error_embed,
    create_chat_embed, create_plan_review_embed,
)
from discord_ui.buttons import ImageActionView, PlanReviewView

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def create_bot() -> commands.Bot:
    intents = discord.Intents.default()
    intents.message_content = True
    intents.messages = True
    return commands.Bot(command_prefix="!", intents=intents)


# ═══════════════════════════════════════════════════════════════
# Bot setup
# ═══════════════════════════════════════════════════════════════

config = load_config()
bot = create_bot()
engine = AIDirectorEngine(config)


# ═══════════════════════════════════════════════════════════════
# Event handlers
# ═══════════════════════════════════════════════════════════════

@bot.event
async def on_ready():
    logger.info(f"Bot logged in as {bot.user} (ID: {bot.user.id})")
    setup_commands(bot, engine)
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} application commands")
    except Exception as e:
        logger.error(f"Command sync failed: {e}")


@bot.event
async def on_message(message: discord.Message):
    """Handle @mentions for conversational AI."""
    if message.author.bot:
        return

    # Process commands first
    await bot.process_commands(message)

    # Check if bot was mentioned
    if bot.user not in message.mentions:
        return

    # Clean the message (remove the mention)
    clean_content = message.content
    for mention in message.mentions:
        clean_content = clean_content.replace(f"<@{mention.id}>", "").replace(f"<@!{mention.id}>", "")
    clean_content = clean_content.strip()

    if not clean_content:
        await message.reply("Hey! 👋 I'm your AI art assistant. Tell me what you'd like to create, or use `/generate` for direct image generation!")
        return

    # Extract image attachments
    images = await _extract_attachments(message)

    async with message.channel.typing():
        try:
            response = await engine.chat(
                message=clean_content,
                user_id=str(message.author.id),
                channel=message.channel,
                images=images if images else None,
            )

            # Build the reply
            reply_content = response.message

            # Add follow-up question formatting
            if response.questions:
                reply_content += "\n\n"
                for i, q in enumerate(response.questions, 1):
                    reply_content += f"**{i}.** {q}\n"

            # If the AI needs an image
            if response.needs_image:
                reply_content += "\n\n📎 *Attach an image to your next message and I'll use it!*"

            embed = create_chat_embed(reply_content, str(message.author.id))

            if response.should_generate:
                embed.set_footer(text="🎨 Generating your image...")

            await message.reply(embed=embed)

        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            await message.reply("❌ Something went wrong. Try again or use `/generate`!")


async def _extract_attachments(message: discord.Message) -> list:
    """Download image attachments from a Discord message."""
    images = []
    for attachment in message.attachments:
        if any(attachment.filename.lower().endswith(ext) for ext in ('.png', '.jpg', '.jpeg', '.webp', '.gif')):
            try:
                img_bytes = await attachment.read()
                images.append(img_bytes)
                logger.info(f"Downloaded attachment: {attachment.filename} ({len(img_bytes)} bytes)")
            except Exception as e:
                logger.warning(f"Failed to download attachment {attachment.filename}: {e}")
    return images


# ═══════════════════════════════════════════════════════════════
# Run the bot
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    token = config.get("discord", {}).get("token", "")
    if not token:
        print("ERROR: No Discord token in config.yaml")
        exit(1)
    bot.run(token)
