# corpus/normalizers/__init__.py

from .tiny_stories_normalizer import TinyStoriesNormalizer
from .wild_chat_normalizer import WildChatNormalizer


__all__ = [
    'TinyStoriesNormalizer',
    'WildChatNormalizer',
]