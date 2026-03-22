"""
Discord Slash Commands — /generate, /workflows, /help, /queue, /status, /upload_workflow.
"""
import io
import json
import logging
import random

import discord
from discord import app_commands

from discord_ui.embeds import (
    create_generating_embed, create_result_embed, create_error_embed,
    create_workflow_list_embed, create_help_embed, create_plan_review_embed,
    create_queue_embed, create_status_embed,
)
from discord_ui.buttons import ImageActionView, PlanReviewView

logger = logging.getLogger(__name__)


def setup_commands(bot, engine):
    """Register all slash commands."""

    @bot.tree.command(name="generate", description="Generate an image from a text prompt")
    @app_commands.describe(
        prompt="What you want to generate",
        workflow="Specific workflow to use (optional)",
        image="Attach a reference image (optional)",
        image2="Attach a second image (optional)",
    )
    async def generate(
        interaction: discord.Interaction,
        prompt: str,
        workflow: str = "",
        image: discord.Attachment = None,
        image2: discord.Attachment = None,
    ):
        await interaction.response.defer()

        # Download image attachments
        images = []
        for att in [image, image2]:
            if att and any(att.filename.lower().endswith(ext) for ext in ('.png', '.jpg', '.jpeg', '.webp')):
                try:
                    img_bytes = await att.read()
                    images.append(img_bytes)
                except Exception as e:
                    logger.warning(f"Failed to read attachment: {e}")

        # Send "generating" status
        embed = create_generating_embed(prompt, workflow_name=workflow or "auto-select")
        status_msg = await interaction.followup.send(embed=embed, wait=True)

        try:
            result_images = await engine.generate(
                prompt=prompt,
                user_id=str(interaction.user.id),
                channel=interaction.channel,
                workflow_override=workflow,
                images=images if images else None,
            )

            if result_images:
                for i, img_data in enumerate(result_images):
                    file = discord.File(io.BytesIO(img_data), filename=f"generated_{i}.png")
                    embed = create_result_embed(
                        prompt=prompt,
                        workflow=workflow or "auto-selected",
                        seed=random.randint(0, 2**31),
                    )
                    view = ImageActionView(engine=engine, prompt=prompt, user_id=str(interaction.user.id))
                    await interaction.channel.send(file=file, embed=embed, view=view)

                # Delete the status message
                try:
                    await status_msg.delete()
                except Exception:
                    pass
            else:
                embed = create_error_embed("Generation failed — no images returned. Try a different prompt or workflow.")
                await status_msg.edit(embed=embed)

        except Exception as e:
            logger.error(f"Generate command failed: {e}", exc_info=True)
            embed = create_error_embed(f"Error: {str(e)[:500]}")
            await status_msg.edit(embed=embed)

    @generate.autocomplete("workflow")
    async def workflow_autocomplete(interaction: discord.Interaction, current: str):
        workflows = engine.get_workflows()
        choices = []
        for name in workflows:
            if current.lower() in name.lower() or not current:
                choices.append(app_commands.Choice(name=name[:100], value=name))
                if len(choices) >= 25:
                    break
        return choices

    @bot.tree.command(name="workflows", description="List available generation workflows")
    async def workflows(interaction: discord.Interaction):
        await interaction.response.defer()
        profiles = engine.get_workflow_profiles()
        workflow_descs = engine.get_workflows()
        embed = create_workflow_list_embed(workflow_descs, profiles)
        await interaction.followup.send(embed=embed)

    @bot.tree.command(name="help", description="How to use the AI Director bot")
    async def help_cmd(interaction: discord.Interaction):
        embed = create_help_embed()
        await interaction.response.send_message(embed=embed)

    @bot.tree.command(name="queue", description="Check the generation queue status")
    async def queue_cmd(interaction: discord.Interaction):
        embed = create_queue_embed()
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @bot.tree.command(name="status", description="Check bot and system status")
    async def status_cmd(interaction: discord.Interaction):
        llm_name = getattr(engine.llm, 'active_provider_name', type(engine.llm).__name__)
        workflow_count = len(engine.workflow_manager.templates)
        embed = create_status_embed(llm_name, workflow_count, engine.config)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @bot.tree.command(name="upload_workflow", description="Upload a new workflow (admin only)")
    @app_commands.describe(
        name="Name for the workflow (no spaces, use underscores)",
        workflow_json="The workflow JSON file",
        description="Description of what this workflow does",
    )
    async def upload_workflow(
        interaction: discord.Interaction,
        name: str,
        workflow_json: discord.Attachment,
        description: str = "",
    ):
        # Check admin permissions
        admin_users = engine.config.get("admin_users", [])
        if admin_users and str(interaction.user.id) not in [str(a) for a in admin_users]:
            await interaction.response.send_message("❌ Only admins can upload workflows.", ephemeral=True)
            return

        await interaction.response.defer()

        try:
            # Read and parse the JSON
            raw_data = await workflow_json.read()
            json_data = json.loads(raw_data.decode("utf-8"))

            # Clean the name
            clean_name = name.replace(" ", "_").lower()

            # Add to workflow manager
            profile = engine.workflow_manager.add_workflow(clean_name, json_data, description)

            if not description:
                prompt = (
                    f"Write a 1-2 sentence description for a ComfyUI workflow named '{clean_name}'. "
                    f"It has architecture '{profile.architecture}', requires {profile.min_images} images, "
                    f"and has capabilities: {', '.join(profile.capabilities)}. "
                    f"Describe its purpose clearly so an AI agent knows when to select it. "
                    f"Return ONLY the description text, no quotes or prefix."
                )
                try:
                    resp = await engine.llm.chat(message=prompt, conversation_history=[], workflows={})
                    description = resp.message.strip()
                    engine.workflow_manager.update_description(clean_name, description)
                except Exception as e:
                    description = f"Autogenerated description failed: {e}"

            class EditDescView(discord.ui.View):
                def __init__(self, wf_name: str, desc: str):
                    super().__init__(timeout=None)
                    self.wf_name = wf_name
                    self.desc = desc

                @discord.ui.button(label="Edit Description", style=discord.ButtonStyle.secondary)
                async def edit_desc(self, interaction: discord.Interaction, button: discord.ui.Button):
                    class DescModal(discord.ui.Modal, title="Edit Workflow Description"):
                        desc_input = discord.ui.TextInput(
                            label="Workflow Description",
                            style=discord.TextStyle.paragraph,
                            default=self.desc,
                            max_length=500,
                            required=True
                        )
                        async def on_submit(self, modal_interaction: discord.Interaction):
                            engine.workflow_manager.update_description(self.wf_name, self.desc_input.value)
                            
                            new_embed = discord.Embed(
                                title="✅ Workflow Uploaded!",
                                description=f"**{self.wf_name}** is now available.\n\n**Description:**\n{self.desc_input.value}",
                                color=0x22c55e,
                            )
                            prof = engine.workflow_manager.profiles.get(self.wf_name)
                            if prof:
                                new_embed.add_field(name="Architecture", value=prof.architecture, inline=True)
                                new_embed.add_field(name="Capabilities", value=", ".join(sorted(prof.capabilities)) or "general", inline=True)
                                new_embed.add_field(name="Image Inputs", value=str(prof.min_images), inline=True)
                                new_embed.add_field(name="Total Nodes", value=str(prof.total_nodes), inline=True)

                            await modal_interaction.response.edit_message(embed=new_embed)
                            self._view.desc = self.desc_input.value

                    modal = DescModal()
                    modal._view = self
                    await interaction.response.send_modal(modal)

            embed = discord.Embed(
                title="✅ Workflow Uploaded!",
                description=f"**{clean_name}** is now available.\n\n**Description:**\n{description}",
                color=0x22c55e,
            )
            embed.add_field(name="Architecture", value=profile.architecture, inline=True)
            embed.add_field(name="Capabilities", value=", ".join(sorted(profile.capabilities)) or "general", inline=True)
            embed.add_field(name="Image Inputs", value=str(profile.min_images), inline=True)
            embed.add_field(name="Total Nodes", value=str(profile.total_nodes), inline=True)

            await interaction.followup.send(embed=embed, view=EditDescView(clean_name, description))
        except json.JSONDecodeError:
            await interaction.followup.send("❌ Invalid JSON file.", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Upload failed: {str(e)[:300]}", ephemeral=True)
