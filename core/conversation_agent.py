"""
Conversation Agent — multi-turn agentic conversation with state machine.

Manages the flow from user intent → questions → plan review → generation.
The agent proactively asks questions, shows plans for review, and handles
the full interactive generation flow.
"""
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """States for the conversation flow."""
    IDLE = "idle"                  # No active flow
    GATHERING_INFO = "gathering"   # Asking questions, collecting details
    REVIEWING_PLAN = "reviewing"   # Showing plan for user confirmation
    GENERATING = "generating"      # Generation in progress
    DONE = "done"                  # Generation complete, awaiting next action


@dataclass
class ConversationContext:
    """Per-user per-channel conversation context."""
    state: ConversationState = ConversationState.IDLE
    user_id: str = ""
    channel_id: str = ""
    
    # Accumulated info
    original_prompt: str = ""
    collected_images: List[bytes] = field(default_factory=list)
    style_preference: str = ""
    aspect_preference: str = ""
    workflow_preference: str = ""
    
    # Pending questions
    pending_questions: List[str] = field(default_factory=list)
    
    # Current plan (if in review state)
    current_plan: Any = None
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    
    def is_stale(self, timeout: float = 300) -> bool:
        """Check if conversation has gone stale (5 min default)."""
        return time.time() - self.last_active > timeout
    
    def touch(self):
        """Update last active timestamp."""
        self.last_active = time.time()


class ConversationAgent:
    """
    Multi-turn conversation agent with state management.
    
    Manages conversation contexts per user/channel and provides
    the logic for when to ask questions, when to show plan review,
    and when to trigger generation.
    """
    
    def __init__(self):
        # Key: f"{user_id}:{channel_id}" -> ConversationContext
        self.contexts: Dict[str, ConversationContext] = {}
    
    def get_context(self, user_id: str, channel_id: str) -> ConversationContext:
        """Get or create conversation context."""
        key = f"{user_id}:{channel_id}"
        
        if key in self.contexts:
            ctx = self.contexts[key]
            if ctx.is_stale():
                # Reset stale context
                ctx = ConversationContext(user_id=user_id, channel_id=channel_id)
                self.contexts[key] = ctx
            else:
                ctx.touch()
            return ctx
        
        ctx = ConversationContext(user_id=user_id, channel_id=channel_id)
        self.contexts[key] = ctx
        return ctx
    
    def reset_context(self, user_id: str, channel_id: str):
        """Reset conversation context to idle."""
        key = f"{user_id}:{channel_id}"
        self.contexts[key] = ConversationContext(user_id=user_id, channel_id=channel_id)
    
    def should_ask_questions(self, ctx: ConversationContext, plan: Any = None, profile: Any = None) -> List[str]:
        """
        Determine if we should ask the user questions before generating.
        
        Returns a list of questions to ask, or empty list if ready to generate.
        """
        questions = []
        
        # If workflow needs images and none provided
        if profile and profile.requires_image and not ctx.collected_images:
            img_purposes = [img.purpose for img in profile.image_inputs]
            purpose_str = ", ".join(img_purposes)
            questions.append(f"📎 This workflow needs {profile.min_images} image(s) ({purpose_str}). Please attach them!")
        
        # If prompt is very short (less than 10 words), suggest more detail
        if ctx.original_prompt and len(ctx.original_prompt.split()) < 5:
            questions.append("✏️ Your prompt is quite short. Want to add more details (style, mood, lighting)?")
        
        return questions
    
    def determine_action(self, ctx: ConversationContext, has_images: bool = False) -> str:
        """
        Determine what action to take based on conversation state.
        
        Returns: 'ask_questions', 'show_plan', 'generate', 'chat'
        """
        if ctx.state == ConversationState.IDLE:
            return 'chat'
        
        if ctx.state == ConversationState.GATHERING_INFO:
            if ctx.pending_questions:
                return 'ask_questions'
            return 'show_plan'
        
        if ctx.state == ConversationState.REVIEWING_PLAN:
            return 'show_plan'
        
        if ctx.state == ConversationState.GENERATING:
            return 'wait'  # Already generating
        
        return 'chat'
    
    def cleanup_stale(self, timeout: float = 600):
        """Remove stale conversation contexts."""
        stale_keys = [
            key for key, ctx in self.contexts.items()
            if ctx.is_stale(timeout)
        ]
        for key in stale_keys:
            del self.contexts[key]
        
        if stale_keys:
            logger.debug(f"Cleaned up {len(stale_keys)} stale conversation contexts")
