"""
User Memory — stores preferences and history for personalization.
"""
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """Persistent profile for a user."""
    user_id: str
    preferred_styles: Dict[str, int] = field(default_factory=dict)
    preferred_models: Dict[str, int] = field(default_factory=dict)
    favorite_prompts: List[str] = field(default_factory=list)
    generation_count: int = 0
    last_used_preset: str = ""
    custom_defaults: Dict[str, Any] = field(default_factory=dict)

    def add_generation(self, style: str, model: str):
        """Update metrics for a new generation."""
        self.generation_count += 1
        
        if style:
            self.preferred_styles[style] = self.preferred_styles.get(style, 0) + 1
            
        if model:
            self.preferred_models[model] = self.preferred_models.get(model, 0) + 1

    def get_top_style(self) -> str:
        """Get the user's most used style."""
        if not self.preferred_styles:
            return "realistic"
        return max(self.preferred_styles.items(), key=lambda x: x[1])[0]

    def get_top_styles(self, n: int = 3) -> List[tuple]:
        """Get the user's top n styles and their counts."""
        if not self.preferred_styles:
            return []
        return sorted(self.preferred_styles.items(), key=lambda x: x[1], reverse=True)[:n]

    def get_top_models(self, n: int = 3) -> List[tuple]:
        """Get the user's top n models and their counts."""
        if not self.preferred_models:
            return []
        return sorted(self.preferred_models.items(), key=lambda x: x[1], reverse=True)[:n]


class UserMemory:
    """Manages persistent user profiles."""

    def __init__(self, config: dict):
        base_dir = Path(__file__).parent.parent
        self.data_dir = base_dir / "data" / "users"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, UserProfile] = {}

    def get_profile(self, user_id: str) -> UserProfile:
        """Get a user's profile, loading from disk if necessary."""
        if user_id in self._cache:
            return self._cache[user_id]

        file_path = self.data_dir / f"{user_id}.json"
        
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                profile = UserProfile(**data)
                self._cache[user_id] = profile
                return profile
            except Exception as e:
                logger.error(f"Failed to load user profile for {user_id}: {e}")
        
        # Create new profile
        profile = UserProfile(user_id=user_id)
        self._cache[user_id] = profile
        return profile

    def save_profile(self, profile: UserProfile):
        """Save a user profile to disk."""
        self._cache[profile.user_id] = profile
        
        file_path = self.data_dir / f"{profile.user_id}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(asdict(profile), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save user profile for {profile.user_id}: {e}")

    def record_generation(self, user_id: str, plan):
        """Record a successful generation to update user preferences."""
        profile = self.get_profile(user_id)
        profile.add_generation(plan.style_category, plan.checkpoint)
        self.save_profile(profile)

    def get_preferences_for_llm(self, user_id: str) -> str:
        """Format user preferences to include in LLM system prompt context."""
        profile = self.get_profile(user_id)
        
        if profile.generation_count == 0:
            return "This is a new user."
            
        lines = [f"User Generation Count: {profile.generation_count}"]
        
        if profile.preferred_styles:
            top_styles = sorted(profile.preferred_styles.items(), key=lambda x: x[1], reverse=True)[:3]
            styles_str = ", ".join([f"{s} ({c} times)" for s, c in top_styles])
            lines.append(f"Favorite Styles: {styles_str}")
            
        if profile.preferred_models:
            top_models = sorted(profile.preferred_models.items(), key=lambda x: x[1], reverse=True)[:3]
            models_str = ", ".join([f"{m.split('.')[0]} ({c} times)" for m, c in top_models])
            lines.append(f"Favorite Models: {models_str}")
            
        return "\n".join(lines)
