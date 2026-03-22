"""
Ollama LLM Provider — local AI models via Ollama.

Supports text models (qwen2.5:7b, llama3.1, etc.) and vision models (llava).
"""
import json
import base64
import logging
from typing import Optional

import aiohttp

from .base import (
    LLMProvider,
    GenerationPlan,
    StyleAnalysis,
    register_provider,
)

logger = logging.getLogger(__name__)

# System prompt for intent analysis — tells the LLM to output structured JSON
INTENT_SYSTEM_PROMPT = """You are an AI Director for an advanced image generation system using ComfyUI.

Your job: analyze the user's prompt and decide the BEST generation strategy.

You will receive:
1. The user's prompt
2. A list of available models, LoRAs, and PRE-BUILT WORKFLOW TEMPLATES.
3. Historical Workflow Performance Scores (if available)

IMPORTANT:
- Focus first and foremost on selecting the right WORKFLOW TEMPLATE.
- We no longer dynamically build graphs node-by-node. You MUST select a workflow template.

You MUST output ONLY valid JSON with this exact structure:
{
    "action": "generate",
    "workflow_template": "name_of_template_here",
    "style_category": "realistic",
    "checkpoint": "model_filename",
    "model_arch": "sdxl",
    "loras": [{"name": "filename.safetensors", "weight": 0.8}],
    "enhanced_prompt": "detailed prompt here...",
    "negative_prompt": "worst quality, low quality...",
    "steps": 30,
    "cfg": 7.0,
    "sampler": "dpmpp_2m_sde",
    "scheduler": "karras",
    "width": 1024,
    "height": 1024,
    "reasoning": "Brief explanation, specifically why you picked this workflow template."
}

Rules:
- ONLY use models/LoRAs/Templates that appear in the available lists. Never invent filenames.
- WORKFLOW TEMPLATES (CRITICAL): You MUST choose `workflow_template` from the "AVAILABLE WORKFLOW TEMPLATES" list. 
- Look at the "Historical Workflow Performance Scores" when choosing a template if provided. If a template has a high score for this type of request, prefer it. Occasionally you can experiment with others to find better results.
- "model_arch" MUST match the model type (sdxl, flux, hunyuan).
- ARCHITECTURE-SPECIFIC SETTINGS (very important):
  - SDXL: cfg 6-8, steps 25-40, sampler dpmpp_2m_sde, scheduler karras
  - Flux Dev: cfg 1.0, steps 20, sampler euler, scheduler simple
  - Flux Turbo/Schnell/Lightning: cfg 1.0, steps 4, sampler euler, scheduler simple
  - Hunyuan: cfg 1.0, steps 30, sampler dpmpp_2m, scheduler normal
- LORA USAGE (EXTREMELY IMPORTANT):
  - ONLY add LoRAs if they match the specific artistic style requested.
  - LoRAs MUST be compatible with the chosen "model_arch".
- enhanced_prompt should be detailed (50-80 words) with lighting, camera, mood, medium details.
- Output ONLY the JSON. No explanations, no markdown."""

ENHANCE_SYSTEM_PROMPT = """You are an expert Stable Diffusion prompt engineer.
The user will provide a basic prompt, and optionally art style info, visual keywords, and LoRA trigger words.
Your task is to rewrite the prompt into a detailed, high-quality Stable Diffusion XL prompt.
Focus on lighting, camera angles, medium, mood, and high quality descriptors.
If LoRA trigger words are provided, you MUST include them exactly as given.
If art style info is provided, match the style closely in your output.
Do NOT include conversational text. Output ONLY the final prompt.
Keep it under 70 words."""

VISION_ANALYSIS_PROMPT = """Analyze this image for an AI image generation system.
Provide a concise response with exactly these sections:
STYLE: (art style in 5-10 words, e.g. 'ethereal Indian oil painting with golden tones')
KEYWORDS: (comma-separated list of 10-15 visual keywords for style matching)
LORA_SEARCH: (2-3 short search terms to find matching LoRA models)
Do NOT include any other text."""

CHAT_SYSTEM_PROMPT = """You are a friendly AI art assistant. You help users create amazing images using Stable Diffusion workflows.
You can discuss art styles, help refine prompts, explain what workflows/models/LoRAs do, and suggest improvements.

AUTONOMOUS GENERATION:
If a user asks you to create or generate an image, and you have enough information, you should trigger a generation by including the tag `<generate>image description</generate>` at the end of your message. 
The text inside the tag should be a descriptive prompt for the image.
Example: "Sure! I'll create a majestic lion for you. <generate>a majestic lion sitting on a rock, sunset, 8k, highly detailed</generate>"

CRITICAL RULES FOR GENERATION:
- Do NOT use the `<generate>` tag if the user's request is vague or missing key details (like style, vibe, subject matter).
- If the request is vague, ask 1 or 2 clarifying questions to strengthen the prompt BEFORE generating.
- Keep responses under 150 words."""

REFINE_SYSTEM_PROMPT = """You are an AI Director refining an image generation plan based on user feedback.

You have an existing generation plan (JSON) and the user wants changes.
Apply the user's feedback to modify the plan, keeping everything else the same.
Only change what the user asks for. Output the complete updated JSON plan.

Rules:
- ONLY use models/LoRAs from the available_models list
- Respect model-LoRA compatibility
- Output ONLY valid JSON, no other text"""


@register_provider("ollama")
class OllamaProvider(LLMProvider):
    """Local LLM via Ollama HTTP API."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = config.get("url", "http://127.0.0.1:11434")
        self.model = config.get("model", "llama3.2:3b")  # Best lightweight model for 6GB VRAM
        self.vision_model = config.get("vision_model", "llava")

    async def _chat(
        self,
        messages: list,
        model: str = None,
        temperature: float = 0.7,
    ) -> str:
        """Send a chat request to Ollama."""
        model = model or self.model
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data["message"]["content"].strip()
        except aiohttp.ClientError as e:
            logger.error(f"Ollama request failed: {e}")
            raise ConnectionError(f"Failed to connect to Ollama at {self.base_url}: {e}")

    def _format_available_models(self, available_models: dict) -> str:
        """Format available models dict into a readable string for the LLM."""
        lines = []
        
        if available_models.get("checkpoints"):
            lines.append("AVAILABLE CHECKPOINTS:")
            for ckpt in available_models["checkpoints"]:
                meta = ""
                if isinstance(ckpt, dict):
                    meta = f" (styles: {', '.join(ckpt.get('styles', []))})"
                    lines.append(f"  - {ckpt['filename']}{meta}")
                else:
                    lines.append(f"  - {ckpt}")
        
        if available_models.get("diffusion_models"):
            lines.append("\nAVAILABLE DIFFUSION MODELS (Flux, Hunyuan, etc.):")
            for dm in available_models["diffusion_models"]:
                meta = ""
                arch = ""
                if isinstance(dm, dict):
                    if dm.get("arch"):
                        arch = f" [{dm['arch'].upper()}]"
                    if dm.get("styles"):
                        meta = f" (styles: {', '.join(dm['styles'])})"
                    lines.append(f"  - {dm['filename']}{arch}{meta}")
                else:
                    lines.append(f"  - {dm}")
        
        if available_models.get("loras"):
            lines.append("\nAVAILABLE LORAS:")
            for lora in available_models["loras"]:
                if isinstance(lora, dict):
                    compat = ""
                    if lora.get("compatible_models"):
                        compat = f" [compatible with: {', '.join(lora['compatible_models'])}]"
                    keywords = ", ".join(lora.get("keywords", []))
                    triggers = ", ".join(lora.get("trigger_words", []))
                    lines.append(
                        f"  - {lora['filename']} (keywords: {keywords})"
                        f" (triggers: {triggers}){compat}"
                    )
                else:
                    lines.append(f"  - {lora}")
        
        if available_models.get("controlnets"):
            lines.append("\nAVAILABLE CONTROLNETS:")
            for cn in available_models["controlnets"]:
                if isinstance(cn, dict):
                    lines.append(f"  - {cn['filename']} (type: {cn.get('type', 'unknown')})")
                else:
                    lines.append(f"  - {cn}")
        
        if available_models.get("ipadapters"):
            lines.append("\nAVAILABLE IP-ADAPTERS:")
            for ipa in available_models["ipadapters"]:
                if isinstance(ipa, dict):
                    lines.append(f"  - {ipa['filename']} (type: {ipa.get('type', 'standard')})")
                else:
                    lines.append(f"  - {ipa}")
        
        return "\n".join(lines)

    def _parse_plan_json(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks."""
        # Try to find JSON in code blocks first
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        # Clean up common issues
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find any JSON object in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Failed to parse LLM JSON output: {text[:200]}")
            return {}

    async def analyze_intent(
        self,
        prompt: str,
        available_models: dict,
        conversation_history: list = None,
        has_reference_image: bool = False,
    ) -> GenerationPlan:
        """Analyze user prompt and build a generation plan."""
        models_str = self._format_available_models(available_models)
        
        user_msg = f"User Prompt: {prompt}\n\n"
        user_msg += f"Reference Image Attached: {'Yes' if has_reference_image else 'No'}\n\n"
        user_msg += f"{models_str}"
        
        messages = [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        ]
        
        # Add conversation history for multi-turn context
        if conversation_history:
            for msg in conversation_history[-6:]:  # Last 6 messages max
                messages.append(msg)
        
        messages.append({"role": "user", "content": user_msg})
        
        response = await self._chat(messages, temperature=0.3)
        plan_dict = self._parse_plan_json(response)
        
        if not plan_dict:
            # Fallback — use basic defaults with enhanced prompt
            logger.warning("Intent analysis returned no valid JSON, using defaults")
            enhanced = await self.enhance_prompt(prompt)
            return GenerationPlan(
                enhanced_prompt=enhanced,
                reasoning="Fallback: could not parse AI analysis, using safe defaults",
            )
        
        # Build GenerationPlan from parsed JSON
        plan = GenerationPlan(
            action=plan_dict.get("action", "generate"),
            style_category=plan_dict.get("style_category", "realistic"),
            checkpoint=plan_dict.get("checkpoint", ""),
            model_arch=plan_dict.get("model_arch", "sdxl"),
            loras=plan_dict.get("loras", []),
            use_controlnet=plan_dict.get("use_controlnet", False),
            controlnet_type=plan_dict.get("controlnet_type", ""),
            controlnet_strength=plan_dict.get("controlnet_strength", 1.0),
            use_ipadapter=plan_dict.get("use_ipadapter", False),
            ipadapter_weight=plan_dict.get("ipadapter_weight", 0.6),
            enhanced_prompt=plan_dict.get("enhanced_prompt", prompt),
            negative_prompt=plan_dict.get("negative_prompt", GenerationPlan.negative_prompt),
            steps=plan_dict.get("steps", 30),
            cfg=plan_dict.get("cfg", 7.0),
            sampler=plan_dict.get("sampler", "dpmpp_2m_sde"),
            scheduler=plan_dict.get("scheduler", "karras"),
            width=plan_dict.get("width", 1024),
            height=plan_dict.get("height", 1024),
            reasoning=plan_dict.get("reasoning", ""),
        )
        
        return plan

    async def enhance_prompt(
        self,
        prompt: str,
        style_info: Optional[str] = None,
        lora_trigger_words: list = None,
    ) -> str:
        """Enhance a basic prompt into a detailed SD prompt."""
        user_content = f"Original Prompt: {prompt}\n"
        if style_info:
            user_content += f"Art Style: {style_info}\n"
        if lora_trigger_words:
            user_content += f"LoRA Trigger Words (MUST include exactly): {', '.join(lora_trigger_words)}\n"
        
        messages = [
            {"role": "system", "content": ENHANCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        
        return await self._chat(messages, temperature=0.7)

    async def analyze_image(self, image_path: str) -> StyleAnalysis:
        """Analyze an image using Ollama vision model."""
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        messages = [
            {
                "role": "user",
                "content": VISION_ANALYSIS_PROMPT,
                "images": [image_b64],
            }
        ]
        
        result = await self._chat(messages, model=self.vision_model)
        
        analysis = StyleAnalysis(raw=result)
        for line in result.split("\n"):
            line = line.strip()
            if line.upper().startswith("STYLE:"):
                analysis.style = line.split(":", 1)[1].strip()
            elif line.upper().startswith("KEYWORDS:"):
                analysis.keywords = line.split(":", 1)[1].strip()
            elif line.upper().startswith("LORA_SEARCH:"):
                terms = line.split(":", 1)[1].strip()
                analysis.lora_search = [t.strip() for t in terms.split(",")]
        
        return analysis

    async def chat(
        self,
        message: str,
        conversation_history: list = None,
        system_context: str = None,
    ) -> str:
        """General conversational chat."""
        messages = [
            {"role": "system", "content": system_context or CHAT_SYSTEM_PROMPT},
        ]
        
        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append(msg)
        
        messages.append({"role": "user", "content": message})
        
        return await self._chat(messages, temperature=0.8)

    async def refine_plan(
        self,
        plan: GenerationPlan,
        user_feedback: str,
        available_models: dict,
    ) -> GenerationPlan:
        """Refine an existing plan based on user feedback."""
        import dataclasses
        plan_dict = dataclasses.asdict(plan)
        models_str = self._format_available_models(available_models)
        
        user_msg = (
            f"Current plan:\n```json\n{json.dumps(plan_dict, indent=2)}\n```\n\n"
            f"User feedback: {user_feedback}\n\n"
            f"{models_str}"
        )
        
        messages = [
            {"role": "system", "content": REFINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        
        response = await self._chat(messages, temperature=0.3)
        refined = self._parse_plan_json(response)
        
        if refined:
            # Update only the fields that changed
            for key, value in refined.items():
                if hasattr(plan, key):
                    setattr(plan, key, value)
        
        return plan
