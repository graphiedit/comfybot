"""
Quality Analyzer — automatically evaluates generated images.

Uses the existing LLM vision provider to detect flaws (hands, anatomy, blur)
and assigns a 1-10 quality score with recommended refinement actions.
"""
import base64
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict

logger = logging.getLogger(__name__)

@dataclass
class QualityScore:
    """The result of a quality analysis."""
    overall: float = 8.0              # 0.0 to 10.0
    faces: float = 8.0
    hands: float = 8.0
    composition: float = 8.0
    sharpness: float = 8.0
    artifacts: float = 2.0            # 0.0 = no artifacts, 10.0 = terrible artifacts
    issues: List[str] = field(default_factory=list)
    raw_response: str = ""

    @property
    def is_acceptable(self) -> bool:
        """Returns True if the image passes the quality threshold."""
        return self.overall >= 6.5 and self.artifacts <= 5.0

    def needs_face_fix(self) -> bool:
        return self.faces < 6.5

    def needs_upscale(self) -> bool:
        return self.sharpness < 6.0


QUALITY_SYSTEM_PROMPT = """You are an expert AI image quality evaluator.
Analyze this generated image for technical flaws, bad anatomy, and overall quality.

You MUST respond with ONLY valid JSON strictly matching this structure:
{
    "overall": 8.5,           // 0-10 score (10 = masterpiece)
    "faces": 9.0,             // 0-10 score (if no faces visible, use 10.0)
    "hands": 8.0,             // 0-10 score (if no hands visible, use 10.0)
    "composition": 8.5,       // 0-10 score
    "sharpness": 8.0,         // 0-10 score
    "artifacts": 1.0,         // 0-10 score FOR BAD ARTIFACTS (0 = none, 10 = terrible ai artifacts)
    "issues": [               // List of strings describing specific flaws
        "slightly blurry background",
        "extra finger on left hand"
    ]
}

Be harsh but fair. AI images often struggle with hands, faces, and extra limbs.
Output ONLY the JSON object. Do not explain your reasoning outside the JSON."""


class QualityAnalyzer:
    """Evaluates images using the LLM provider to determine if they need refinement."""

    def __init__(self, llm_provider):
        self.llm = llm_provider

    async def analyze(self, image_bytes: bytes) -> QualityScore:
        """Analyze an image's quality using vision AI."""
        # Convert bytes to base64
        import tempfile
        import os
        
        tmp_path = os.path.join(tempfile.gettempdir(), f"qa_temp.png")
        try:
            with open(tmp_path, "wb") as f:
                f.write(image_bytes)
                
            # Unfortunately our LLMProvider base class only exposes analyze_image(path)
            # which returns a StyleAnalysis. We need a general vision chat.
            # But the Ollama/Gemini providers have chat functionality we can leverage or 
            # we can hack it by using the LLM provider's internal methods depending on the provider.
            
            # Since the interface for raw vision chat isn't standardized across providers,
            # we will attempt to extract what we need based on the provider type.
            
            if hasattr(self.llm, "api_key") and self.llm.__class__.__name__ == "GeminiProvider":
                score = await self._analyze_gemini(tmp_path)
            elif hasattr(self.llm, "vision_model") and self.llm.__class__.__name__ == "OllamaProvider":
                score = await self._analyze_ollama(tmp_path, image_bytes)
            else:
                logger.warning(f"Unknown LLM provider {self.llm.__class__.__name__} for QualityAnalyzer. Returning default score.")
                score = QualityScore()
                
            return score
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return QualityScore(overall=10.0)  # Pass-through on error
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
                
    async def _analyze_gemini(self, image_path: str) -> QualityScore:
        """Analyze with Gemini."""
        from google.genai import types
        try:
            image_file = self.llm.client.files.upload(file=image_path)
            
            response = await self.llm.client.aio.models.generate_content(
                model=self.llm.model_name,
                contents=[
                    image_file, 
                    QUALITY_SYSTEM_PROMPT
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                )
            )
            
            return self._parse_json_result(response.text)
        except Exception as e:
            logger.error(f"Gemini Quality Analysis failed: {e}")
            raise
            
    async def _analyze_ollama(self, image_path: str, image_bytes: bytes) -> QualityScore:
        """Analyze with Ollama Llava."""
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        messages = [
            {
                "role": "user",
                "content": QUALITY_SYSTEM_PROMPT,
                "images": [image_b64],
            }
        ]
        
        response = await self.llm._chat(messages, model=self.llm.vision_model, temperature=0.1)
        return self._parse_json_result(response)
        
    def _parse_json_result(self, text: str) -> QualityScore:
        """Parse the JSON output into a QualityScore object."""
        # Clean up code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
            
        text = text.strip()
        
        try:
            # Try to find JSON object bounds if there's extra text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return QualityScore(
                    overall=float(data.get("overall", 8.0)),
                    faces=float(data.get("faces", 8.0)),
                    hands=float(data.get("hands", 8.0)),
                    composition=float(data.get("composition", 8.0)),
                    sharpness=float(data.get("sharpness", 8.0)),
                    artifacts=float(data.get("artifacts", 2.0)),
                    issues=data.get("issues", []),
                    raw_response=text
                )
        except Exception as e:
            logger.error(f"Failed to parse quality analysis JSON: {e}\nResponse: {text[:200]}")
            
        return QualityScore(raw_response=text)

    def suggest_refinement(self, score: QualityScore) -> Dict:
        """Suggest an action to fix the image based on its scores."""
        if score.is_acceptable:
            return {"action": "none"}
            
        if score.needs_face_fix():
            return {"action": "face_fix", "reason": "Low face quality detected"}
            
        if score.needs_upscale():
            return {"action": "upscale", "reason": "Image lacks sharpness/detail"}
            
        if score.artifacts > 7.0 or score.hands < 5.0:
            return {"action": "regenerate", "reason": "Severe anatomical issues or artifacts"}
            
        return {"action": "refine", "reason": "Overall quality could be improved"}
