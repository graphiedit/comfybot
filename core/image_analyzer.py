"""
Image Analyzer — advanced vision analysis for tool detection.

Interprets the content of a reference image to decide which controlnets
or ip-adapters would work best.
"""
import base64
import json
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ImageAnalysis:
    """Detailed structural analysis of a reference image."""
    has_face: bool = False
    has_pose: bool = False      # Full body/limbs visible
    has_depth: bool = False     # Strong depth/3D structure
    has_edges: bool = False     # Strong lineart/contours
    style_info: str = ""
    recommended_tools: List[str] = field(default_factory=list) # e.g. ["openpose", "ipadapter"]
    raw_response: str = ""


IMAGE_ANALYZER_PROMPT = """Analyze this image structurally to help guide an AI generation pipeline.
We need to know what tools (ControlNet/IP-Adapter) would be best to extract reference data.

Provide ONLY a valid JSON object matching exactly this format:
{
    "has_face": true,            // Is there a clear, prominent human face?
    "has_pose": true,            // Is a human body/pose clearly visible?
    "has_depth": true,           // Is there strong 3D structure or perspective?
    "has_edges": false,          // Is this lineart, sketch, or has strong outlines?
    "style_info": "realistic lighting, oil painting texture",
    "recommended_tools": [       // List any of: "openpose", "depth", "canny", "lineart", "ipadapter", "faceid"
        "openpose", 
        "ipadapter"
    ]
}"""


class ImageAnalyzer:
    """Uses LLM vision to extract structural meaning from reference images."""
    
    def __init__(self, llm_provider):
        self.llm = llm_provider

    async def analyze(self, image_bytes: bytes) -> ImageAnalysis:
        """Perform structural analysis of an image."""
        try:
            # Hacky way to use vision capabilities across varying providers
            import tempfile
            import os
            
            tmp_path = os.path.join(tempfile.gettempdir(), f"img_analyzer_temp.png")
            try:
                with open(tmp_path, "wb") as f:
                    f.write(image_bytes)
                
                if hasattr(self.llm, "api_key") and self.llm.__class__.__name__ == "GeminiProvider":
                    from google.genai import types
                    image_file = self.llm.client.files.upload(file=tmp_path)
                    
                    response = await self.llm.client.aio.models.generate_content(
                        model=self.llm.model_name,
                        contents=[image_file, IMAGE_ANALYZER_PROMPT],
                        config=types.GenerateContentConfig(response_mime_type="application/json")
                    )
                    text = response.text
                
                elif hasattr(self.llm, "vision_model") and self.llm.__class__.__name__ == "OllamaProvider":
                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                    messages = [{"role": "user", "content": IMAGE_ANALYZER_PROMPT, "images": [image_b64]}]
                    text = await self.llm._chat(messages, model=self.llm.vision_model)
                
                else:
                    return ImageAnalysis()

                return self._parse(text)
                
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return ImageAnalysis()

    def _parse(self, text: str) -> ImageAnalysis:
        """Parse the JSON output."""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
            
        text = text.strip()
        
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return ImageAnalysis(
                    has_face=data.get("has_face", False),
                    has_pose=data.get("has_pose", False),
                    has_depth=data.get("has_depth", False),
                    has_edges=data.get("has_edges", False),
                    style_info=data.get("style_info", ""),
                    recommended_tools=data.get("recommended_tools", []),
                    raw_response=text
                )
        except Exception:
            pass
            
        return ImageAnalysis(raw_response=text)
