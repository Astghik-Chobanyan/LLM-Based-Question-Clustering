from pathlib import Path

from llama_index.core.prompts import PromptTemplate

from utils.singleton_meta import SingletonMeta


class PromptTemplates(metaclass=SingletonMeta):
    """Singleton for loading and retrieving prompt templates by name and type."""

    def __init__(self):
        self.prompt_templates_path = Path(__file__).resolve().parent / "templates"
        if not self.prompt_templates_path.exists():
            raise ValueError(f"Prompt path not found: {self.prompt_templates_path}.")

        # Structure: { "prompt_name": { "user": PromptTemplate, "system": PromptTemplate } }
        self.prompt_templates = {}
        self._load_prompt_templates()

    def _load_prompt_templates(self):
        """Load all prompt templates (.user and .system) from the templates directory."""
        for prompt_file in self.prompt_templates_path.rglob("*.*"):
            if prompt_file.suffix not in [".user", ".system"]:
                continue

            prompt_name = prompt_file.stem 
            prompt_type = prompt_file.suffix.lstrip(".")  # "user" or "system"

            # Initialize entry if not already
            if prompt_name not in self.prompt_templates:
                self.prompt_templates[prompt_name] = {}

            # Load the prompt text
            self.prompt_templates[prompt_name][prompt_type] = self._load_prompt_template_text(prompt_file)

    def _load_prompt_template_text(self, prompt_file: Path) -> PromptTemplate:
        """Read a prompt template file and wrap it with PromptTemplate."""
        with open(prompt_file, "r", encoding="utf-8") as f:
            return PromptTemplate(f.read())

    def get_prompt_template(self, name: str, prompt_type: str):
        """Retrieve a specific template by name and type (user/system)."""
        if name not in self.prompt_templates:
            raise ValueError(f"Prompt '{name}' not found.")
        if prompt_type not in self.prompt_templates[name]:
            raise ValueError(f"No '{prompt_type}' template found for '{name}'.")
        return self.prompt_templates[name][prompt_type]

    def get_prompt(self, name: str, prompt_type: str, **kwargs) -> str:
        """Format a prompt template with arguments."""
        template = self.get_prompt_template(name, prompt_type)
        return template.format(**kwargs)

    def list_prompts(self):
        """Return available prompt names and their types."""
        return {
            name: list(types.keys())
            for name, types in self.prompt_templates.items()
        }
