"""Prompt templates for LLM-based question clustering."""

import sys
from pathlib import Path

# Add parent directory to path to allow sibling imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from prompts.prompt_templates import PromptTemplates

__all__ = ['PromptTemplates']


