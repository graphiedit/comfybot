"""
Ollama LLM Provider — local AI brain via Ollama.

Optimized for small models (qwen2.5:1.5b). Uses extremely structured
prompts to get reliable JSON output from limited-capacity models.
"""
import json
import re
import random
import logging
from typing import Optional, List, Dict

import aiohttp

from llm.base import LLMProvider, GenerationPlan, ChatResponse

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# SYSTEM PROMPTS — Carefully tuned for small models
# ═══════════════════════════════════════════════════════════════

INTENT_SYSTEM_PROMPT = """You are an AI image generation assistant. Your job is to:
1. Pick the BEST workflow template for the user's request
2. Write a detailed image generation prompt
3. Determine if the user needs to provide images

AVAILABLE WORKFLOW TEMPLATES (with capabilities):
{workflow_list}

RULES:
- You MUST pick a workflow_template from the list above
- Pick the template whose capabilities and description best match the request
- If the user wants to EDIT an image, pick a workflow with 'image_edit' capability
- If the user has NO image attached, pick a workflow that does NOT require image input
- If a workflow requires images and the user hasn't provided any, set "needs_image" to true
- Write an enhanced_prompt with 40-80 words: include subject, style, lighting, colors, camera angle, mood
- If user wants a specific size, set width and height (default 1024x1024)
- For landscape: width=1344, height=768. For portrait: width=768, height=1344

OUTPUT FORMAT — respond with ONLY this JSON, nothing else:
{{"workflow_template": "template_name", "enhanced_prompt": "detailed prompt...", "negative_prompt": "worst quality, low quality, bad anatomy, bad hands, text, watermark", "width": 1024, "height": 1024, "needs_image": false, "reasoning": "why you picked this template"}}"""

CHAT_SYSTEM_PROMPT = """You are a creative AI art assistant called the AI Director. You help artists create amazing images.

YOUR IDENTITY:
- You are friendly, enthusiastic, and knowledgeable about art and image generation
- You run on a local AI system connected to ComfyUI
- You can generate images directly from conversation
- You understand art styles, composition, lighting, and photography

AVAILABLE WORKFLOWS (with capabilities and image requirements):
{workflow_list}

CAPABILITIES:
- Generate images from text descriptions
- Edit or restyle existing images (when user provides an image)
- Discuss art techniques, styles, and composition
- Help users refine their creative ideas
- Suggest improvements to prompts

HOW TO GENERATE IMAGES:
When a user asks you to create/generate/make an image, and you have enough detail:
1. End your message with: <generate>detailed image prompt here</generate>
2. The prompt inside <generate> tags should be 40-80 words with style, lighting, mood details
3. Also specify the workflow if relevant: <workflow>template_name</workflow>

IMPORTANT AGENTIC BEHAVIOR:
- If the request is vague, ask 1-2 quick questions to get better results (style? mood? aspect ratio?)
- If a workflow needs an image input and user hasn't provided one, ask them to attach an image
- Proactively suggest improvements: "Want me to try this in landscape format?" or "I can also try a different style"
- After generating, ask if they want variations or adjustments
- Use <questions>your questions here</questions> tags when asking follow-up questions

CONVERSATION RULES:
- Keep responses under 120 words
- Be warm and encouraging
- Use emoji occasionally 🎨 ✨ 🖌️
- If user shares an idea, help them develop it
- Never refuse a creative request — help make it work
- Don't be overly technical unless asked"""

ENHANCE_SYSTEM_PROMPT = """You are an expert prompt engineer for AI image generation.
Rewrite the user's image prompt to be more detailed and vivid for Stable Diffusion / Flux image generation.
Add specific details about: lighting, camera angle, mood, medium, art style, color palette.
Keep the enhanced prompt between 40-80 words. Output ONLY the enhanced prompt, nothing else.
If LoRA trigger words are provided, include them EXACTLY as-is in the enhanced prompt."""

VISION_ANALYSIS_PROMPT = """Analyze this image and describe:
STYLE: The art style (e.g., photorealistic, anime, oil painting, digital art, etc.)
KEYWORDS: Key visual elements as comma-separated keywords
LORA_SEARCH: Suggest 2-3 search terms for finding similar LoRAs

Format your response exactly as:
STYLE: <style description>
KEYWORDS: <keyword1, keyword2, ...>
LORA_SEARCH: <term1, term2, term3>"""

REFINE_SYSTEM_PROMPT = """You are an AI image generation planner. The user wants to modify their generation plan.
You will receive the current plan as JSON and user feedback.
Apply the user's requested changes to the plan and return the COMPLETE updated plan as JSON.
Only modify the fields the user explicitly asks to change. Keep everything else the same.
Return ONLY valid JSON, nothing else."""


ENHANCE_PROMPT = """Rewrite this image prompt to be more detailed and vivid for AI image generation.
Add: lighting, camera angle, mood, medium, style details.
Keep it under 70 words. Output ONLY the enhanced prompt, nothing else.

Original: {prompt}"""


class OllamaProvider(LLMProvider):
    """Local LLM via Ollama HTTP API — optimized for small models."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = config.get("url", "http://127.0.0.1:11434")
        self.model = config.get("model", "qwen2.5:1.5b")
        self.vision_model = config.get("vision_model", "llava")
        logger.info(f"Ollama provider: {self.base_url} model={self.model}")

    async def _call_ollama(
        self,
        prompt: str,
        system: str = "",
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """Make a raw call to Ollama API."""
        model = model or self.model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        raise RuntimeError(f"Ollama error {resp.status}: {error[:200]}")
                    data = await resp.json()
                    return data.get("response", "").strip()
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Cannot connect to Ollama at {self.base_url}: {e}")

    def _format_workflow_list(self, workflows: Dict[str, str]) -> str:
        """Format workflows into a clear list for the LLM."""
        if not workflows:
            return "No workflows available."
        
        lines = []
        for name, description in workflows.items():
            lines.append(f"- {name}: {description}")
        return "\n".join(lines)

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from LLM response — handles messy outputs."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in the text
        # Look for { ... } blocks
        brace_depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if brace_depth == 0:
                    start = i
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0 and start >= 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        start = -1
        
        # Try removing markdown code fences
        cleaned = re.sub(r'```json\s*', '', text)
        cleaned = re.sub(r'```\s*', '', cleaned)
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            pass
        
        logger.warning(f"Could not parse JSON from LLM response: {text[:200]}")
        return {}

    async def analyze_intent(
        self,
        prompt: str,
        workflows: Dict[str, str],
        user_history: Optional[List[str]] = None,
    ) -> GenerationPlan:
        """Analyze user intent and pick the best workflow."""
        workflow_list = self._format_workflow_list(workflows)
        system = INTENT_SYSTEM_PROMPT.format(workflow_list=workflow_list)
        
        # Add context about what they've generated before
        context = ""
        if user_history:
            recent = user_history[-3:]
            context = f"\nUser's recent prompts: {', '.join(recent)}\n"
        
        user_prompt = f"{context}User request: {prompt}"
        
        response = await self._call_ollama(
            prompt=user_prompt,
            system=system,
            temperature=0.4,  # Lower temp for more reliable JSON
            max_tokens=400,
        )
        
        data = self._parse_json_response(response)
        
        if not data:
            # Fallback: use first available workflow with the raw prompt
            logger.warning("LLM returned no valid JSON, using fallback")
            fallback_template = next(iter(workflows), "")
            return GenerationPlan(
                workflow_template=fallback_template,
                enhanced_prompt=prompt,
                reasoning="Fallback: could not parse AI response",
            )
        
        # Validate workflow_template exists
        template = data.get("workflow_template", "")
        if template not in workflows:
            # Find closest match
            template_lower = template.lower()
            for wf_name in workflows:
                if template_lower in wf_name.lower() or wf_name.lower() in template_lower:
                    template = wf_name
                    break
            else:
                # Just use first available
                template = next(iter(workflows), "")
                logger.warning(f"LLM picked unknown template '{data.get('workflow_template')}', using {template}")
        
        return GenerationPlan(
            workflow_template=template,
            enhanced_prompt=data.get("enhanced_prompt", prompt),
            negative_prompt=data.get("negative_prompt", "worst quality, low quality, bad anatomy, bad hands, text, watermark"),
            width=int(data.get("width", 1024)),
            height=int(data.get("height", 1024)),
            reasoning=data.get("reasoning", ""),
        )

    async def enhance_prompt(self, prompt: str, style_hints: str = "") -> str:
        """Enhance a basic prompt into a detailed generation prompt."""
        user_msg = ENHANCE_PROMPT.format(prompt=prompt)
        if style_hints:
            user_msg += f"\nStyle hints: {style_hints}"
        
        response = await self._call_ollama(
            prompt=user_msg,
            temperature=0.8,
            max_tokens=150,
        )
        
        # Clean up — remove quotes if the model wrapped it
        response = response.strip('"\'')
        return response if response else prompt

    async def chat(
        self,
        message: str,
        conversation_history: List[Dict[str, str]],
        workflows: Dict[str, str],
    ) -> ChatResponse:
        """Have a conversation. May trigger image generation."""
        workflow_list = self._format_workflow_list(workflows)
        system = CHAT_SYSTEM_PROMPT.format(workflow_list=workflow_list)
        
        # Build conversation context
        context_parts = []
        for entry in conversation_history[-6:]:  # Last 6 messages
            role = entry.get("role", "user")
            content = entry.get("content", "")
            context_parts.append(f"{role}: {content}")
        
        if context_parts:
            full_prompt = "\n".join(context_parts) + f"\nuser: {message}\nassistant:"
        else:
            full_prompt = f"user: {message}\nassistant:"
        
        response = await self._call_ollama(
            prompt=full_prompt,
            system=system,
            temperature=0.8,
            max_tokens=300,
        )
        
        # Check if the AI wants to generate an image
        generate_match = re.search(r'<generate>(.*?)</generate>', response, re.DOTALL)
        workflow_match = re.search(r'<workflow>(.*?)</workflow>', response, re.DOTALL)
        questions_match = re.search(r'<questions>(.*?)</questions>', response, re.DOTALL)
        
        # Clean the tags from the displayed message
        clean_message = re.sub(r'<generate>.*?</generate>', '', response, flags=re.DOTALL)
        clean_message = re.sub(r'<workflow>.*?</workflow>', '', clean_message, flags=re.DOTALL)
        clean_message = re.sub(r'<questions>.*?</questions>', '', clean_message, flags=re.DOTALL)
        clean_message = clean_message.strip()
        
        result = ChatResponse(message=clean_message)
        
        if generate_match:
            result.should_generate = True
            result.generation_prompt = generate_match.group(1).strip()
            if workflow_match:
                result.workflow_hint = workflow_match.group(1).strip()
        
        if questions_match:
            q_text = questions_match.group(1).strip()
            result.questions = [q.strip() for q in q_text.split('\n') if q.strip()]
        
        return result

