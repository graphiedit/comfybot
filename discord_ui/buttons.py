"""
Discord Buttons — interactive views for image actions and plan review.
"""
import asyncio
import io
import logging
import random

import discord

logger = logging.getLogger(__name__)


class ImageActionView(discord.ui.View):
    """Buttons shown after image generation: Retry, Vary, Wide, Tall, Delete."""

    def __init__(self, engine, prompt: str, user_id: str, timeout: float = 300):
        super().__init__(timeout=timeout)
        self.engine = engine
        self.prompt = prompt
        self.user_id = user_id

    async def _regenerate(self, interaction: discord.Interaction, action: str):
        """Common handler for regeneration actions."""
        await interaction.response.defer()

        from discord_ui.embeds import create_generating_embed, create_result_embed, create_error_embed

        try:
            if action == "retry":
                images = await self.engine.retry_generation(interaction.channel, self.user_id)
            elif action == "vary":
                images = await self.engine.vary_generation(interaction.channel, self.user_id)
            elif action == "wide":
                images = await self.engine.change_aspect(interaction.channel, self.user_id, "wide")
            elif action == "tall":
                images = await self.engine.change_aspect(interaction.channel, self.user_id, "tall")
            else:
                images = []

            if images:
                for i, img_data in enumerate(images):
                    file = discord.File(io.BytesIO(img_data), filename=f"generated_{i}.png")
                    embed = create_result_embed(
                        prompt=self.prompt,
                        workflow=action,
                        seed=random.randint(0, 2**31),
                    )
                    view = ImageActionView(engine=self.engine, prompt=self.prompt, user_id=self.user_id)
                    await interaction.channel.send(file=file, embed=embed, view=view)
            else:
                embed = create_error_embed(f"{action.capitalize()} failed — no images returned.")
                await interaction.followup.send(embed=embed, ephemeral=True)

        except Exception as e:
            logger.error(f"{action} failed: {e}", exc_info=True)
            embed = create_error_embed(f"Error: {str(e)[:300]}")
            await interaction.followup.send(embed=embed, ephemeral=True)

    @discord.ui.button(label="Retry", emoji="🔄", style=discord.ButtonStyle.secondary)
    async def retry(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._regenerate(interaction, "retry")

    @discord.ui.button(label="Vary", emoji="🎲", style=discord.ButtonStyle.secondary)
    async def vary(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._regenerate(interaction, "vary")

    @discord.ui.button(label="Wide", emoji="↔️", style=discord.ButtonStyle.secondary)
    async def wide(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._regenerate(interaction, "wide")

    @discord.ui.button(label="Tall", emoji="↕️", style=discord.ButtonStyle.secondary)
    async def tall(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._regenerate(interaction, "tall")

    @discord.ui.button(label="Delete", emoji="🗑️", style=discord.ButtonStyle.danger)
    async def delete(self, interaction: discord.Interaction, button: discord.ui.Button):
        if str(interaction.user.id) == self.user_id:
            await interaction.message.delete()
        else:
            await interaction.response.send_message("Only the requester can delete this.", ephemeral=True)


class PlanReviewView(discord.ui.View):
    """Buttons for reviewing a generation plan before execution."""

    def __init__(self, engine, plan, user_id: str, channel, images=None, timeout: float = 120):
        super().__init__(timeout=timeout)
        self.engine = engine
        self.plan = plan
        self.user_id = user_id
        self.channel = channel
        self.images = images

    @discord.ui.button(label="Generate", emoji="✅", style=discord.ButtonStyle.success)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Confirm and start generation."""
        if str(interaction.user.id) != self.user_id:
            await interaction.response.send_message("Only the requester can confirm.", ephemeral=True)
            return

        await interaction.response.defer()

        from discord_ui.embeds import create_generating_embed, create_result_embed, create_error_embed

        embed = create_generating_embed(self.plan.enhanced_prompt, self.plan.workflow_template)
        await interaction.message.edit(embed=embed, view=None)

        try:
            result = await self.engine._execute_with_recovery(self.plan, self.channel)
            if result:
                for i, img_data in enumerate(result):
                    file = discord.File(io.BytesIO(img_data), filename=f"generated_{i}.png")
                    embed = create_result_embed(
                        prompt=self.plan.enhanced_prompt,
                        workflow=self.plan.workflow_template,
                        seed=self.plan.seed,
                    )
                    view = ImageActionView(
                        engine=self.engine,
                        prompt=self.plan.enhanced_prompt,
                        user_id=self.user_id,
                    )
                    await self.channel.send(file=file, embed=embed, view=view)
                # Delete the review message
                try:
                    await interaction.message.delete()
                except Exception:
                    pass
            else:
                embed = create_error_embed("Generation failed — no images returned.")
                await interaction.message.edit(embed=embed, view=None)
        except Exception as e:
            embed = create_error_embed(f"Error: {str(e)[:500]}")
            await interaction.message.edit(embed=embed, view=None)

    @discord.ui.button(label="Edit Prompt", emoji="✏️", style=discord.ButtonStyle.primary)
    async def edit_prompt(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Open a modal to edit the prompt."""
        if str(interaction.user.id) != self.user_id:
            await interaction.response.send_message("Only the requester can edit.", ephemeral=True)
            return

        modal = PromptEditModal(self.plan, self)
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="Cancel", emoji="❌", style=discord.ButtonStyle.danger)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Cancel the generation."""
        if str(interaction.user.id) != self.user_id:
            await interaction.response.send_message("Only the requester can cancel.", ephemeral=True)
            return

        embed = discord.Embed(
            title="❌ Generation Cancelled",
            color=0x6b7280,
        )
        await interaction.message.edit(embed=embed, view=None)


class PromptEditModal(discord.ui.Modal, title="Edit Prompt"):
    """Modal for editing the generation prompt before submitting."""

    def __init__(self, plan, review_view: PlanReviewView):
        super().__init__()
        self.plan = plan
        self.review_view = review_view
        self.prompt_input = discord.ui.TextInput(
            label="Enhanced Prompt",
            style=discord.TextStyle.paragraph,
            default=plan.enhanced_prompt[:4000],
            max_length=4000,
            required=True,
        )
        self.add_item(self.prompt_input)

    async def on_submit(self, interaction: discord.Interaction):
        self.plan.enhanced_prompt = self.prompt_input.value

        from discord_ui.embeds import create_plan_review_embed
        embed = create_plan_review_embed(self.plan)
        await interaction.response.edit_message(embed=embed, view=self.review_view)
