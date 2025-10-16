"""
LLM-based clustering of extracted intents into broader categories.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

# Add parent directory to path to allow imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from prompts.prompt_templates import PromptTemplates

load_dotenv()

# Pydantic models for structured LLM output
class Cluster(BaseModel):
    """Assignment of an intent to a cluster."""
    cluster_label: str = Field(description="The cluster label being assigned")
    cluster_description: str = Field(description="A brief description of the cluster")
    intents: List[str] = Field(description="The intents belonging to the cluster")

class Clusters(BaseModel):
    """Result of clustering a batch of intents."""
    clusters: List[Cluster]




class LLMClusterer:
    """Clusters extracted intents into broader categories using LLM."""

    def __init__(self, model_name: str = "gpt-5-mini-2025-08-07", temperature: float = 0.0):
        """Initialize the LLM clusterer.
        
        Args:
            model_name: Name of the OpenAI model to use
            temperature: Temperature for LLM generation (0.0 for deterministic)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.initial_llm = ChatOpenAI(
            model=model_name, 
            temperature=temperature
        ).with_structured_output(Clusters)
        self.batch_llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        ).with_structured_output(Clusters)
        
        self.clusters: Dict[str, Cluster] = {}
        self.prompt_templates = PromptTemplates()

    def load_analysis(self, analysis_path: str) -> List[Cluster]:
        """Load intent analysis.
        
        Args:
            analysis_path: Path to the analysis JSON file
            
        Returns:
            List of intent groups
        """
        with open(analysis_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data['groups']
    
    def cluster_intents(self, intent_groups, batch_size: int = 50) -> Dict:
        """Cluster intents into broader categories using LLM.
        
        Args:
            intent_groups: List of intent groups to cluster
            batch_size: Number of intents to process in each LLM call
            
        Returns:
            Dictionary mapping cluster labels to IntentCluster objects
        """
        print(f"\nClustering {len(intent_groups)} intents into categories...")
        print(f"Processing in batches of {batch_size}")
        
        clusters = sorted(intent_groups, key=lambda x: x['count'], reverse=True)
        
        for i in tqdm(range(0, len(intent_groups), batch_size), desc="Clustering batches"):
            batch = clusters[i:i + batch_size]
            if i == 0:
                self._process_initial_clusters(batch)
            else:
                self._process_batch_clusters(batch)
        
        return self.clusters
    
    def _process_initial_clusters(self, intent_groups):
        """Process a batch of intents and assign them to clusters.
        
        Args:
            intent_groups: Batch of intent groups to process
        """
        intent_list = " \n ".join(group['extracted_intent'] for group in intent_groups)

        
        messages = [
            ('system', self.prompt_templates.get_prompt('initial_clusters', 'system')),
            ('human', self.prompt_templates.get_prompt('initial_clusters', 'user').format(intent_list=intent_list))
        ]
        
        result = self.initial_llm.invoke(messages)
        
        for cluster in result.clusters:
            cluster_label = cluster.cluster_label.lower().strip()
            
            self.clusters[cluster_label] = cluster
    
    def _process_batch_clusters(self, intent_groups):
        """Process a batch of intents and assign them to clusters.
        
        Args:
            intent_groups: Batch of intent groups to process
        """
    
        intent_list = " \n ".join(group['extracted_intent'] for group in intent_groups)

        existing_clusters = " \n ".join(cluster.cluster_label + ": " + cluster.cluster_description for cluster in self.clusters.values())

        messages = [
            ('system', self.prompt_templates.get_prompt('batch_clustering', 'system')),
            ('human', self.prompt_templates.get_prompt('batch_clustering', 'user').format(existing_clusters=existing_clusters, new_intent_list=intent_list))
        ]

        result = self.batch_llm.invoke(messages)

        total_counts_of_intents = 0
        for cluster in result.clusters:
            cluster_label = cluster.cluster_label.lower().strip()
            
            if cluster_label not in self.clusters:
                self.clusters[cluster_label] = cluster
                total_counts_of_intents += len(cluster.intents)
            else:
                self.clusters[cluster_label].intents.extend(list(set(cluster.intents)))
                total_counts_of_intents += len(set(cluster.intents))
        logger.info(f"Total counts of intents: {total_counts_of_intents}, batch size: {len(intent_groups)}")
    
    
    def save_results(self, output_path: str):
        """Save clustering results to JSON file.
        
        Args:
            output_path: Path to save the results
        """
        output_data = {
            'total_clusters': len(self.clusters),
            'clusters': []
        }
        
        for label, cluster in sorted(self.clusters.items()):
            output_data['clusters'].append({
                'cluster_label': label,
                'cluster_description': cluster.cluster_description,
                'intent_count': len(cluster.intents),
                'intents': sorted(cluster.intents)
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Clustering results saved to: {output_path}")
    

    def print_summary(self):
        """Print a summary of the clustering results."""
        print("\n" + "=" * 80)
        print("INTENT CLUSTERING SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal clusters created: {len(self.clusters)}")
        print(f"Total intents clustered: {sum(len(c.intents) for c in self.clusters.values())}")
        
        print("\n" + "-" * 80)
        print("TOP CLUSTERS BY INTENT COUNT")
        print("-" * 80)
        
        sorted_clusters = sorted(
            self.clusters.items(),
            key=lambda x: len(x[1].intents),
            reverse=True
        )
        
        print(f"\n{'Rank':<6} {'Intents':<10} {'Cluster Label':<30} {'Description'}")
        print("-" * 80)
        
        for rank, (label, cluster) in enumerate(sorted_clusters[:20], 1):
            desc = cluster.cluster_description[:50] + '...' if len(cluster.cluster_description) > 50 else cluster.cluster_description
            print(f"{rank:<6} {len(cluster.intents):<10} {label:<30} {desc}")
        
        if len(sorted_clusters) > 20:
            print(f"\n... and {len(sorted_clusters) - 20} more clusters")


def add_questions_to_clusters(data_with_questions_path: str, cluster_data_path: str) -> Dict:
    """Add questions to clusters.
    
    Args:
        data_with_questions_path: Path to the data with questions
        cluster_data_path: Path to the cluster data
        data_with_questions: List of data with questions
    """
    with open(data_with_questions_path, 'r', encoding='utf-8') as f:
        data_with_questions = json.load(f)
    with open(cluster_data_path, 'r', encoding='utf-8') as f:
        cluster_data = json.load(f)

    intent_to_questions = {}
    for group in data_with_questions['groups']:
        intent = group['extracted_intent']
        if intent not in intent_to_questions:
            intent_to_questions[intent] = []
        intent_to_questions[intent].extend(group['all_questions'])

    for cluster in cluster_data['clusters']:
        cluster['questions'] = []
        for intent in cluster['intents']:
            if intent in intent_to_questions:
                cluster['questions'].extend(intent_to_questions[intent])
        cluster['question_count'] = len(cluster['questions'])
    with open(cluster_data_path.replace('.json', '_for_evaluation.json'), 'w', encoding='utf-8') as f:
        json.dump(cluster_data, f, indent=2, ensure_ascii=False)
    return cluster_data

