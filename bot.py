"""
AI Director Bot — Discord bot entry point.

Starts the Discord bot, initializes the AI Director engine,
registers slash commands, and handles the bot lifecycle.
"""
import asyncio
import logging
import sys
from pathlib import Path

import discord
from discord.ext import commands
import yaml

from engine import AIDirectorEngine
from discord_ui.commands import setup_commands

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ai_director")

# Reduce noisy loggers
logging.getLogger("discord").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        logger.error("Please create config.yaml — see README.md for format")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


class AIDirectorBot(commands.Bot):
    """Discord bot with AI Director engine integration."""

    def __init__(self, config: dict):
        intents = discord.Intents.default()
        intents.message_content = True
        
        super().__init__(
            command_prefix="!",
            intents=intents,
            description="🎨 AI Director — Smart Image Generation",
        )
        
        self.config = config
        self.engine = AIDirectorEngine(config)

    async def setup_hook(self):
        """Called when the bot starts up — register commands and start engine."""
        logger.info("Setting up AI Director Bot...")
        
        # Register slash commands
        setup_commands(self)
        
        # Start the engine
        await self.engine.start()
        
        # Sync commands with Discord
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} slash commands")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")

    async def on_ready(self):
        """Called when bot is fully connected to Discord."""
        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"  🎨 AI Director Bot is ONLINE")
        logger.info(f"  Bot: {self.user.name}#{self.user.discriminator}")
        logger.info(f"  Guilds: {len(self.guilds)}")
        logger.info(f"  LLM: {self.config.get('llm', {}).get('provider', 'ollama')}")
        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Set rich presence
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="for /generate • AI Director",
            )
        )

    async def on_message(self, message: discord.Message):
        """Handle regular messages — enables conversational mode."""
        if message.author.bot:
            return
        
        # If the bot is mentioned, treat it as a chat
        if self.user and self.user.mentioned_in(message):
            # Remove mention from message
            clean_content = message.content.replace(f"<@{self.user.id}>", "").strip()
            clean_content = clean_content.replace(f"<@!{self.user.id}>", "").strip()
            
            # If nothing left, just return
            if not clean_content:
                await self.process_commands(message)
                return
                
            logger.info(f"Received mention chat text: {clean_content}")
            
            async with message.channel.typing():
                try:
                    response = await self.engine.chat(
                        user_id=str(message.author.id),
                        message=clean_content,
                    )
                    
                    from discord_ui.embeds import create_chat_embed
                    embed = create_chat_embed(response)
                    await message.reply(embed=embed)
                except Exception as e:
                    logger.error(f"Chat error: {e}", exc_info=True)
                    await message.reply(f"❌ Sorry, I ran into an error: {str(e)[:200]}")
        
        await self.process_commands(message)

    async def close(self):
        """Clean shutdown."""
        logger.info("Shutting down AI Director...")
        await self.engine.stop()
        await super().close()


def main():
    """Entry point."""
    config = load_config()
    
    # Get Discord token
    token = config.get("discord", {}).get("token", "")
    
    if not token or token == "YOUR_DISCORD_BOT_TOKEN":
        logger.error("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.error("  ❌ Discord bot token not configured!")
        logger.error("")
        logger.error("  Add your bot token to config.yaml:")
        logger.error("    discord:")
        logger.error("      token: 'YOUR_BOT_TOKEN_HERE'")
        logger.error("")
        logger.error("  Get a token at: https://discord.com/developers/applications")
        logger.error("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        sys.exit(1)
    
    bot = AIDirectorBot(config)
    
    try:
        bot.run(token, log_handler=None)
    except discord.LoginFailure:
        logger.error("Invalid Discord token! Check your config.yaml")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")


if __name__ == "__main__":
    main()
