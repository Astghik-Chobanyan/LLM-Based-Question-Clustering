from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd


@dataclass
class QuestionDataPoint:
    """Data class to hold a single question with its metadata."""
    question: str  # The instruction/question text
    category: str  # The category label
    flags: str     # Additional flags from the dataset
    intent: str    # The intent label

class QuestionDataLoader:
    """Data loader for question clustering task."""
    
    def __init__(self, data_path: str):
        """Initialize the data loader.
        
        Args:
            data_path: Path to the CSV file containing the dataset
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        
        # Load the dataset
        self.df = pd.read_csv(self.data_path)
        
        # Validate required columns
        required_columns = ['instruction', 'category', 'flags', 'intent']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def get_all_data(self, shuffle: bool = False, random_state: Optional[int] = None) -> List[QuestionDataPoint]:
        """Get all data points from the dataset.
        
        Args:
            shuffle: Whether to shuffle the data randomly
            random_state: Random state for reproducibility (only used if shuffle=True)
        
        Returns:
            List of QuestionDataPoint objects
        """
        df = self.df
        
        # Shuffle if requested
        if shuffle:
            df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        
        return [
            QuestionDataPoint(
                question=row['instruction'],
                category=row['category'],
                flags=row['flags'],
                intent=row['intent']
            )
            for _, row in df.iterrows()
        ]
    
    def get_data_by_category(self, category: str) -> List[QuestionDataPoint]:
        """Get all data points for a specific category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of QuestionDataPoint objects for the specified category
        """
        category_df = self.df[self.df['category'] == category]
        return [
            QuestionDataPoint(
                question=row['instruction'],
                category=row['category'],
                flags=row['flags'],
                intent=row['intent']
            )
            for _, row in category_df.iterrows()
        ]
    
    def get_unique_categories(self) -> List[str]:
        """Get list of unique categories in the dataset.
        
        Returns:
            List of category names
        """
        return self.df['category'].unique().tolist()
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of data points across categories.
        
        Returns:
            Dictionary mapping category names to counts
        """
        return self.df['category'].value_counts().to_dict()

