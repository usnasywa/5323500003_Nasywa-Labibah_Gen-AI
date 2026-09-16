# my_ai_project/__init__.py
from .config.llm_config import LLMConfig
from .conversation.history import ConversationHistory

__all__ = ["LLMConfig", "ConversationHistory"]