"""
Google Gemini LLM Provider — cloud AI models via google-genai.

Supports text and vision models via Gemini API.
"""
import json
import logging
from typing import Optional, List, Dict, Any

from google import genai
from google.genai import types

from .base import (
    LLMProvider,
    GenerationPlan,
    ChatResponse,
    register_provider,
)

logger = logging.getLogger(__name__)

# Re-use the excellent system prompts from Ollama provider
from .ollama_provider import (
    INTENT_SYSTEM_PROMPT,
    ENHANCE_SYSTEM_PROMPT,
    VISION_ANALYSIS_PROMPT,
    CHAT_SYSTEM_PROMPT,
    REFINE_SYSTEM_PROMPT,
)


@register_provider("gemini")
class GeminiProvider(LLMProvider):
    """Google Gemini provider — requires API key configuration."""

    def __init__(self, config: dict):
        super().__init__(config)
        
        # In the config structure, gemini config is under config["gemini"]
        # when initialized via create_llm_provider
        self.api_key = config.get("api_key", "")
        self.model_name = config.get("model", "gemini-2.5-flash")
        
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            logger.warning(
                "Gemini provider created but no API key configured. "
                "Set llm.gemini.api_key in config.yaml"
            )

    def _check_configured(self):
        if not self.client:
            raise RuntimeError(
                "Gemini is not configured. Add your API key to config.yaml:\n"
                "  llm:\n"
                "    gemini:\n"
                "      api_key: 'YOUR_KEY_HERE'"
            )

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
        
        if available_models.get("workflow_templates"):
            lines.append("\nAVAILABLE WORKFLOW TEMPLATES (You MUST choose one of these):")
            lines.append(available_models["workflow_templates"])
            
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

    def _convert_history(self, history: Optional[List[Dict[str, Any]]]) -> List[types.Content]:
        """Convert standard format [{'role': 'user', 'content': '...'}] to Gemini format."""
        if not history:
            return []
            
        gemini_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
        return gemini_history

    async def analyze_intent(
        self,
        prompt: str,
        available_models: dict,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        has_reference_image: bool = False,
    ) -> GenerationPlan:
        self._check_configured()
        
        models_str = self._format_available_models(available_models)
        
        user_msg = f"User Prompt: {prompt}\n\n"
        user_msg += f"Reference Image Attached: {'Yes' if has_reference_image else 'No'}\n\n"
        user_msg += f"{models_str}"
        
        history = None
        if conversation_history:
            history = self._convert_history(conversation_history[-6:])
        
        # For latest google-genai, the AsyncClient is available via client.aio
        chat = self.client.aio.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=INTENT_SYSTEM_PROMPT,
                temperature=0.3,
                response_mime_type="application/json",
            ),
            history=history,
        )
        
        response = await chat.send_message(user_msg)
        plan_dict = self._parse_plan_json(response.text)
        
        if not plan_dict:
            # Fallback — use basic defaults with enhanced prompt
            logger.warning("Intent analysis returned no valid JSON, using defaults")
            try:
                enhanced = await self.enhance_prompt(prompt)
            except Exception:
                enhanced = prompt
            return GenerationPlan(
                enhanced_prompt=enhanced,
                reasoning="Fallback: could not parse AI analysis, using safe defaults",
            )
        
        # Build GenerationPlan from parsed JSON
        plan = GenerationPlan(
            action=plan_dict.get("action", "generate"),
            workflow_template=plan_dict.get("workflow_template", ""),
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
        lora_trigger_words: Optional[List[str]] = None,
    ) -> str:
        self._check_configured()
        
        user_content = f"Original Prompt: {prompt}\n"
        if style_info:
            user_content += f"Art Style: {style_info}\n"
        if lora_trigger_words:
            user_content += f"LoRA Trigger Words (MUST include exactly): {', '.join(lora_trigger_words)}\n"
        
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=ENHANCE_SYSTEM_PROMPT,
                temperature=0.7,
            )
        )
        return response.text.strip()

    async def analyze_image(self, image_path: str) -> dict:
        self._check_configured()
        
        # Use the File API from the genai client
        image_file = self.client.files.upload(file=image_path)
        
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=[
                image_file, 
                VISION_ANALYSIS_PROMPT
            ]
        )
        
        result = response.text
        analysis = {"raw": result, "style": "", "keywords": "", "lora_search": []}
        
        for line in result.split("\n"):
            line = line.strip()
            if line.upper().startswith("STYLE:"):
                analysis["style"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("KEYWORDS:"):
                analysis["keywords"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("LORA_SEARCH:"):
                terms = line.split(":", 1)[1].strip()
                analysis["lora_search"] = [t.strip() for t in terms.split(",")]
        
        return analysis

    async def chat(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        system_context: Optional[str] = None,
        workflows: Optional[dict] = None,
    ) -> ChatResponse:
        self._check_configured()
        
        import re
        
        # Format workflow list for system context
        workflow_list = ""
        if workflows:
            workflow_list = "\n".join(f"- {name}: {desc}" for name, desc in workflows.items())
        
        system = CHAT_SYSTEM_PROMPT.format(workflow_list=workflow_list or "None loaded")
        if system_context:
            system += f"\n\n{system_context}"
        
        history = None
        if conversation_history:
            history = self._convert_history(conversation_history[-10:])
            
        chat_session = self.client.aio.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.8,
            ),
            history=history,
        )
        
        response = await chat_session.send_message(message)
        text = response.text.strip()
        
        # Parse generation trigger
        generate_match = re.search(r'<generate>(.*?)</generate>', text, re.DOTALL)
        workflow_match = re.search(r'<workflow>(.*?)</workflow>', text, re.DOTALL)
        questions_match = re.search(r'<questions>(.*?)</questions>', text, re.DOTALL)
        
        # Clean tags from display text
        clean = re.sub(r'<generate>.*?</generate>', '', text, flags=re.DOTALL)
        clean = re.sub(r'<workflow>.*?</workflow>', '', clean, flags=re.DOTALL)
        clean = re.sub(r'<questions>.*?</questions>', '', clean, flags=re.DOTALL)
        clean = clean.strip()
        
        result = ChatResponse(message=clean)
        
        if generate_match:
            result.should_generate = True
            result.generation_prompt = generate_match.group(1).strip()
            if workflow_match:
                result.workflow_hint = workflow_match.group(1).strip()
        
        if questions_match:
            q_text = questions_match.group(1).strip()
            result.questions = [q.strip() for q in q_text.split('\n') if q.strip()]
        
        return result

    async def refine_plan(
        self,
        plan: GenerationPlan,
        user_feedback: str,
        available_models: dict,
    ) -> GenerationPlan:
        self._check_configured()
        
        import dataclasses
        plan_dict = dataclasses.asdict(plan)
        models_str = self._format_available_models(available_models)
        
        user_msg = (
            f"Current plan:\n```json\n{json.dumps(plan_dict, indent=2)}\n```\n\n"
            f"User feedback: {user_feedback}\n\n"
            f"{models_str}"
        )
        
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=REFINE_SYSTEM_PROMPT,
                temperature=0.3,
                response_mime_type="application/json",
            )
        )
        refined = self._parse_plan_json(response.text)
        
        if refined:
            # Update only the fields that changed
            for key, value in refined.items():
                if hasattr(plan, key):
                    setattr(plan, key, value)
            
        return plan
