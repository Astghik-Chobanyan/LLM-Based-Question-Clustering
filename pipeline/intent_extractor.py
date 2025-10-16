import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from tqdm import tqdm

# Add parent directory to path to allow imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from pipeline.data_loader import QuestionDataPoint
from prompts.prompt_templates import PromptTemplates



class QuestionIntent(BaseModel):
    """Pydantic model for structured intent extraction."""
    intent: str = Field(description="The primary intent or purpose of the question")


@dataclass
class IntentExtraction:
    """Data class to hold a question with its extracted intent."""
    question: str
    category: str  # Original category from dataset
    extracted_intent: str
    flags: str
    original_intent: str  # Original intent from dataset


class IntentExtractor:
    """Extract intents from questions using LLM."""

    def __init__(self, model_name: str = "qwen3:8b", temperature: float = 0.0):
        """Initialize the intent extractor.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Temperature for generation (0.0 for deterministic)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature
        ).with_structured_output(QuestionIntent)
        
    def extract_intent(self, question: str) -> str:
        """Extract intent from a single question.
        
        Args:
            question: The question text
            
        Returns:
            The extracted intent
        """
        prompt_templates = PromptTemplates()
        system_prompt = prompt_templates.get_prompt_template('intent_extractor', 'system')
        user_prompt = prompt_templates.get_prompt_template('intent_extractor', 'user').format(question=question)
        result = self.llm.invoke([
            ('system', system_prompt),
            ('user', user_prompt)
        ])
        return result.intent
    
    def extract_intents_batch(
        self, 
        data_points: List[QuestionDataPoint],
        show_progress: bool = True,
        checkpoint_file: Optional[str] = None,
        checkpoint_interval: int = 100
    ) -> List[IntentExtraction]:
        """Extract intents from a batch of questions with checkpointing.
        
        Args:
            data_points: List of QuestionDataPoint objects
            show_progress: Whether to show progress bar
            checkpoint_file: Path to save checkpoint results (optional)
            checkpoint_interval: Save checkpoint after this many questions (default: 100)
            
        Returns:
            List of IntentExtraction objects
        """
        results = []
        
        # Load existing results if checkpoint file exists
        if checkpoint_file and Path(checkpoint_file).exists():
            print(f"Loading existing results from checkpoint: {checkpoint_file}")
            results = self.load_results(checkpoint_file)
            print(f"Loaded {len(results)} existing results. Continuing from there...")
        
        # Skip already processed questions
        start_idx = len(results)
        remaining_data = data_points[start_idx:]
        
        if not remaining_data:
            print("All questions already processed!")
            return results
        
        print(f"Processing {len(remaining_data)} remaining questions...")
        
        iterator = tqdm(
            enumerate(remaining_data, start=start_idx), 
            desc="Extracting intents",
            total=len(remaining_data),
            initial=0
        ) if show_progress else enumerate(remaining_data, start=start_idx)
        
        for idx, data_point in iterator:
            try:
                intent = self.extract_intent(data_point.question)
                results.append(
                    IntentExtraction(
                        question=data_point.question,
                        category=data_point.category,
                        extracted_intent=intent,
                        flags=data_point.flags,
                        original_intent=data_point.intent
                    )
                )
            except Exception as e:
                print(f"\nError extracting intent for question: {data_point.question[:50]}...")
                print(f"Error: {e}")
                # Add a placeholder for failed extractions
                results.append(
                    IntentExtraction(
                        question=data_point.question,
                        category=data_point.category,
                        extracted_intent="ERROR: Failed to extract",
                        flags=data_point.flags,
                        original_intent=data_point.intent
                    )
                )
            
            # Save checkpoint at intervals
            if checkpoint_file and (idx + 1) % checkpoint_interval == 0:
                self.save_results(results, checkpoint_file)
                print(f"\n✓ Checkpoint saved at {idx + 1} questions")
        
        if checkpoint_file:
            self.save_results(results, checkpoint_file)
            print(f"\n✓ Final results saved: {len(results)} total extractions")
        
        return results
    
    def save_results(self, results: List[IntentExtraction], output_path: str):
        """Save extraction results to a JSON file.
        
        Args:
            results: List of IntentExtraction objects
            output_path: Path to save the results
        """
        data = [
            {
                'question': r.question,
                'category': r.category,
                'extracted_intent': r.extracted_intent,
                'flags': r.flags,
                'original_intent': r.original_intent
            }
            for r in results
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
    
    def load_results(self, input_path: str) -> List[IntentExtraction]:
        """Load extraction results from a JSON file.
        
        Args:
            input_path: Path to the saved results
            
        Returns:
            List of IntentExtraction objects
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [
            IntentExtraction(
                question=item['question'],
                category=item['category'],
                extracted_intent=item['extracted_intent'],
                flags=item['flags'],
                original_intent=item['original_intent']
            )
            for item in data
        ]

