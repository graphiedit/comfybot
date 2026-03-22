# 🎨 AI Director — Smart ComfyUI Image Generation

An intelligent image generation system that uses ComfyUI as a backend with an AI-powered decision-making layer. Instead of using fixed workflows, the system **analyzes your prompt** and **dynamically constructs the best generation pipeline** — choosing the right model, LoRAs, ControlNet, IP-Adapter, and settings automatically.

Think of it as **MidJourney-level simplicity** with **ComfyUI-level flexibility**.

## How It Works

```
User → Discord Bot → AI Director (LLM) → Workflow Builder → ComfyUI → Result
```

1. You send a prompt via Discord (e.g. `/generate a cyberpunk samurai at sunset`)
2. The AI Director (powered by a local LLM) analyzes your prompt:
   - Selects the best checkpoint model for the style
   - Picks matching LoRAs
   - Decides if ControlNet or IP-Adapter should be used
   - Rewrites your prompt into a detailed SD prompt
3. A custom ComfyUI workflow is built dynamically
4. ComfyUI generates the image
5. The result is sent back with **interactive buttons** (retry, upscale, variations)

## Features

- 🧠 **AI-Powered Decisions** — LLM analyzes intent and builds optimal pipelines
- 🔄 **Dynamic Workflows** — no fixed templates, each generation is unique
- 🎯 **Model Awareness** — knows which LoRAs work with which checkpoints
- 🎮 **Discord Interface** — slash commands + interactive buttons
- 💬 **Conversational** — chat with the AI Director, it can ask you questions
- 🔌 **Swappable LLM** — start with Ollama (local), switch to Gemini/OpenAI anytime
- ⬆️ **Post-Generation** — retry, upscale, create variations with one click
- 📊 **Queue System** — handles multiple users with a single GPU

## Discord Commands

| Command | Description |
|---------|-------------|
| `/generate <prompt> [image] [model] [style]` | Generate an image |
| `/upscale <image>` | Upscale an image 2x |
| `/vary <prompt> [strength]` | Create variations |
| `/edit <prompt> <image>` | Edit an image with a new prompt |
| `/chat <message>` | Chat with the AI Director |
| `/models` | List available models, LoRAs, etc. |
| `/queue` | Check generation queue status |
| `/settings` | View default generation settings |

You can also **@mention the bot** in any channel for conversational interaction.

## Setup

### 1. Prerequisites

- **Python 3.10+**
- **ComfyUI** running locally (default: `http://127.0.0.1:8188`)
- **Ollama** with a model pulled (e.g. `ollama pull qwen2.5:7b`)
- **Discord Bot Token** ([create one here](https://discord.com/developers/applications))

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. ComfyUI Custom Nodes (Optional)

Install via ComfyUI Manager for additional features:
- **ComfyUI_IPAdapter_plus** (cubiq) — for style references
- **comfyui_controlnet_aux** (Fannovel16) — for pose/depth control

### 4. Configure

Edit `config.yaml`:

```yaml
discord:
  token: "YOUR_DISCORD_BOT_TOKEN"

llm:
  provider: "ollama"    # or "gemini"
  ollama:
    model: "qwen2.5:7b"
```

### 5. Run

```bash
python bot.py
```

The bot will connect to Discord, sync slash commands, and start listening!

## Switching LLM Providers

The LLM layer is abstracted — change one line in config to switch:

```yaml
llm:
  provider: "gemini"    # Switch from "ollama" to "gemini"
  gemini:
    api_key: "YOUR_GEMINI_KEY"
```

To add a new provider, create a file in `llm/` that subclasses `LLMProvider`.

## Project Structure

```
├── bot.py                  # Discord bot entry point
├── engine.py               # AI Director — central orchestrator
├── config.yaml             # All configuration
├── core/
│   ├── comfyui_client.py   # ComfyUI API communication
│   ├── workflow_builder.py # Dynamic workflow JSON construction
│   └── queue_manager.py    # Multi-user job queue
├── llm/
│   ├── base.py             # Abstract LLM provider + GenerationPlan
│   ├── ollama_provider.py  # Ollama implementation
│   └── gemini_provider.py  # Gemini stub
├── registry/
│   ├── model_registry.py   # Auto-discovers models from ComfyUI
│   └── data/               # Model & LoRA catalogs (YAML)
└── discord_ui/
    ├── commands.py          # Slash commands
    ├── buttons.py           # Interactive buttons
    └── embeds.py            # Rich embeds
```

## Adding Models to the Catalog

Edit `registry/data/model_catalog.yaml` to tell the AI about your models:

```yaml
checkpoints:
  - filename: "Juggernaut-XL_v9_RunDiffusion.safetensors"
    styles: ["realistic", "cinematic", "portrait"]
    quality: 9
    base_model: "sdxl"
```

Edit `registry/data/lora_catalog.yaml` for LoRAs:

```yaml
loras:
  - filename: "cyberpunk_neon_v1.safetensors"
    keywords: ["cyberpunk", "neon", "futuristic"]
    trigger_words: ["cybrpnk style"]
    compatible_models: ["sdxl"]
```
