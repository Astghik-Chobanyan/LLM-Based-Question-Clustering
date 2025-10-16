"""
Evaluate LLM clustering quality by analyzing original category distribution.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_clusters(file_path: str) -> Dict:
    """Load cluster data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_category_distribution(cluster_data: Dict) -> Dict:
    """Calculate category distribution for each cluster.
    
    Returns:
        Dictionary with cluster statistics including category purity
    """
    results = {
        'total_clusters': cluster_data['total_clusters'],
        'total_questions': 0,  # Will calculate from questions
        'cluster_details': [],
        'overall_stats': {
            'avg_purity': 0,
            'avg_entropy': 0,
            'highly_pure_clusters': 0,  # >80% purity
            'moderately_pure_clusters': 0,  # 60-80% purity
            'mixed_clusters': 0,  # <60% purity
        }
    }
    
    purity_scores = []
    
    for cluster in cluster_data['clusters']:
        cluster_label = cluster['cluster_label']
        cluster_description = cluster.get('cluster_description', '')
        
        # Count categories directly from questions
        category_counts = defaultdict(int)
        questions = cluster.get('questions', [])
        total_questions_in_cluster = len(questions)
        
        for question in questions:
            category = question.get('category', 'UNKNOWN')
            category_counts[category] += 1
        
        if total_questions_in_cluster == 0:
            continue
        
        sorted_categories = sorted(
            category_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Calculate purity (percentage of dominant category)
        dominant_category = sorted_categories[0][0]
        dominant_count = sorted_categories[0][1]
        purity = (dominant_count / total_questions_in_cluster) * 100
        purity_scores.append(purity)
        
        # Calculate entropy (measure of disorder)
        entropy = 0
        for count in category_counts.values():
            p = count / total_questions_in_cluster
            if p > 0:
                entropy -= p * (p ** 0.5)  
        
        category_distribution = {}
        for category, count in sorted_categories:
            percentage = (count / total_questions_in_cluster) * 100
            category_distribution[category] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        if purity >= 80:
            purity_level = 'high'
            results['overall_stats']['highly_pure_clusters'] += 1
        elif purity >= 60:
            purity_level = 'moderate'
            results['overall_stats']['moderately_pure_clusters'] += 1
        else:
            purity_level = 'low'
            results['overall_stats']['mixed_clusters'] += 1
        
        cluster_info = {
            'cluster_label': cluster_label,
            'cluster_description': cluster_description,
            'total_questions': total_questions_in_cluster,
            'intent_count': cluster.get('intent_count', len(cluster.get('intents', []))),
            'dominant_category': dominant_category,
            'dominant_count': dominant_count,
            'purity_percentage': round(purity, 2),
            'purity_level': purity_level,
            'category_count': len(category_counts),
            'category_distribution': category_distribution,
            'entropy_score': round(entropy, 3)
        }
        
        results['cluster_details'].append(cluster_info)
        results['total_questions'] += total_questions_in_cluster
    
    if purity_scores:
        results['overall_stats']['avg_purity'] = round(sum(purity_scores) / len(purity_scores), 2)
    
    return results


def print_evaluation_report(results: Dict, top_n: int = None, min_questions: int = 0):
    """Print a formatted evaluation report."""
    
    print("\n" + "=" * 100)
    print("CLUSTER QUALITY EVALUATION REPORT")
    print("=" * 100)
    
    stats = results['overall_stats']
    print(f"\n{'OVERALL STATISTICS':^100}")
    print("-" * 100)
    print(f"Total Clusters: {results['total_clusters']}")
    print(f"Total Questions: {results['total_questions']:,}")
    print(f"Average Purity: {stats['avg_purity']:.2f}%")
    print(f"\nPurity Distribution:")
    print(f"  High Purity (≥80%):     {stats['highly_pure_clusters']:3d} clusters")
    print(f"  Moderate Purity (60-80%): {stats['moderately_pure_clusters']:3d} clusters")
    print(f"  Low Purity (<60%):      {stats['mixed_clusters']:3d} clusters")
    
    clusters = results['cluster_details']
    if min_questions > 0:
        clusters = [c for c in clusters if c['total_questions'] >= min_questions]
    
    clusters_by_purity = sorted(clusters, key=lambda x: x['purity_percentage'], reverse=True)
    
    clusters_by_size = sorted(clusters, key=lambda x: x['total_questions'], reverse=True)
    
    if top_n:
        clusters_by_purity = clusters_by_purity[:top_n]
        clusters_by_size = clusters_by_size[:top_n]
    
    print(f"\n{'CLUSTERS SORTED BY PURITY':^100}")
    print("-" * 100)
    print(f"{'Rank':<6} {'Purity':<8} {'Questions':<10} {'Categories':<11} {'Cluster Label':<30} {'Dominant Category'}")
    print("-" * 100)
    
    for rank, cluster in enumerate(clusters_by_purity, 1):
        purity_str = f"{cluster['purity_percentage']:.1f}%"
        
        if cluster['purity_level'] == 'high':
            purity_str = f"{purity_str} ✓"
        elif cluster['purity_level'] == 'low':
            purity_str = f"{purity_str} ✗"
        
        print(f"{rank:<6} {purity_str:<8} {cluster['total_questions']:<10} "
              f"{cluster['category_count']:<11} {cluster['cluster_label'][:30]:<30} "
              f"{cluster['dominant_category']}")
    
    print(f"\n{'LARGEST CLUSTERS BY QUESTION COUNT':^100}")
    print("-" * 100)
    print(f"{'Rank':<6} {'Questions':<10} {'Purity':<8} {'Intents':<8} {'Cluster Label':<35} {'Dominant Category'}")
    print("-" * 100)
    
    for rank, cluster in enumerate(clusters_by_size, 1):
        purity_str = f"{cluster['purity_percentage']:.1f}%"
        print(f"{rank:<6} {cluster['total_questions']:<10} {purity_str:<8} "
              f"{cluster['intent_count']:<8} {cluster['cluster_label'][:35]:<35} "
              f"{cluster['dominant_category']}")
    
    print(f"\n{'DETAILED CATEGORY DISTRIBUTION (Top 10 Clusters by Size)':^100}")
    print("=" * 100)
    
    for cluster in clusters_by_size[:10]:
        print(f"\n{'Cluster: ' + cluster['cluster_label']:<50} Purity: {cluster['purity_percentage']:.1f}%")
        print(f"Description: {cluster['cluster_description']}")
        print(f"Total Questions: {cluster['total_questions']:,} | Intents: {cluster['intent_count']}")
        print("-" * 100)
        
        for category, data in list(cluster['category_distribution'].items())[:10]:
            bar_length = int(data['percentage'] / 2)  
            bar = '█' * bar_length
            print(f"  {category:<30} {data['count']:>6} ({data['percentage']:>6.2f}%) {bar}")
        
        if len(cluster['category_distribution']) > 10:
            remaining = len(cluster['category_distribution']) - 10
            print(f"  ... and {remaining} more categories")


def save_evaluation_results(results: Dict, output_path: str):
    """Save evaluation results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Evaluation results saved to: {output_path}")


def create_evaluation_plots(results: Dict, output_dir: str):
    """Create and save evaluation plots.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating evaluation plots in '{output_dir}/'...")
    print("-" * 80)
    
    # 1. Purity distribution plot
    plot_purity_distribution(results, output_dir)
    
    # 2. Top clusters by purity
    plot_top_clusters_by_purity(results, output_dir, top_n=20)
    
    # 3. Top clusters by size
    plot_top_clusters_by_size(results, output_dir, top_n=20)
    
    # 4. Purity vs size scatter plot
    plot_purity_vs_size(results, output_dir)
    
    # 5. Category distribution in top clusters
    plot_category_distribution(results, output_dir, top_n=10)
    
    # 6. Overall summary dashboard
    plot_summary_dashboard(results, output_dir)
    
    print("-" * 80)
    print(f"✓ All evaluation plots saved to: {output_dir}/")


def plot_purity_distribution(results: Dict, output_dir: str):
    """Plot distribution of purity scores across clusters."""
    clusters = results['cluster_details']
    purity_scores = [c['purity_percentage'] for c in clusters]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram
    ax1.hist(purity_scores, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(results['overall_stats']['avg_purity'], color='red', 
                linestyle='--', linewidth=2, label=f"Average: {results['overall_stats']['avg_purity']:.1f}%")
    ax1.axvline(80, color='green', linestyle='--', alpha=0.5, label='High Purity Threshold (80%)')
    ax1.axvline(60, color='orange', linestyle='--', alpha=0.5, label='Moderate Purity Threshold (60%)')
    ax1.set_xlabel('Purity Percentage', fontweight='bold')
    ax1.set_ylabel('Number of Clusters', fontweight='bold')
    ax1.set_title('Distribution of Cluster Purity Scores', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(purity_scores, vert=True)
    ax2.axhline(results['overall_stats']['avg_purity'], color='red', 
                linestyle='--', linewidth=2, label=f"Average: {results['overall_stats']['avg_purity']:.1f}%")
    ax2.axhline(80, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(60, color='orange', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Purity Percentage', fontweight='bold')
    ax2.set_title('Purity Score Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'purity_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: purity_distribution.png")
    plt.close()


def plot_top_clusters_by_purity(results: Dict, output_dir: str, top_n: int = 20):
    """Plot top N clusters by purity score."""
    clusters = sorted(results['cluster_details'], key=lambda x: x['purity_percentage'], reverse=True)[:top_n]
    
    labels = [c['cluster_label'][:30] for c in clusters]
    purity = [c['purity_percentage'] for c in clusters]
    colors = ['green' if p >= 80 else 'orange' if p >= 60 else 'red' for p in purity]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, purity, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Purity Percentage', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Clusters by Purity Score', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, p) in enumerate(zip(bars, purity)):
        ax.text(p + 1, i, f'{p:.1f}%', va='center', fontsize=8, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='High Purity (≥80%)'),
        Patch(facecolor='orange', alpha=0.7, label='Moderate Purity (60-80%)'),
        Patch(facecolor='red', alpha=0.7, label='Low Purity (<60%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'top_{top_n}_by_purity.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: top_{top_n}_by_purity.png")
    plt.close()


def plot_top_clusters_by_size(results: Dict, output_dir: str, top_n: int = 20):
    """Plot top N clusters by question count."""
    clusters = sorted(results['cluster_details'], key=lambda x: x['total_questions'], reverse=True)[:top_n]
    
    labels = [c['cluster_label'][:30] for c in clusters]
    sizes = [c['total_questions'] for c in clusters]
    purity = [c['purity_percentage'] for c in clusters]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, sizes, alpha=0.7, edgecolor='black')
    
    for bar, p in zip(bars, purity):
        if p >= 80:
            bar.set_color('green')
        elif p >= 60:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Number of Questions', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Clusters by Question Count\n(Color indicates purity)', 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    
    for i, (bar, size, p) in enumerate(zip(bars, sizes, purity)):
        ax.text(size + max(sizes)*0.01, i, f'{size} ({p:.1f}%)', 
                va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'top_{top_n}_by_size.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: top_{top_n}_by_size.png")
    plt.close()


def plot_purity_vs_size(results: Dict, output_dir: str):
    """Scatter plot of purity vs cluster size."""
    clusters = results['cluster_details']
    
    sizes = [c['total_questions'] for c in clusters]
    purity = [c['purity_percentage'] for c in clusters]
    colors = ['green' if p >= 80 else 'orange' if p >= 60 else 'red' for p in purity]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(sizes, purity, c=colors, s=100, alpha=0.6, edgecolors='black')
    
    ax.axhline(80, color='green', linestyle='--', alpha=0.3, label='High Purity (80%)')
    ax.axhline(60, color='orange', linestyle='--', alpha=0.3, label='Moderate Purity (60%)')
    ax.axhline(results['overall_stats']['avg_purity'], color='red', 
               linestyle='--', alpha=0.5, label=f"Average: {results['overall_stats']['avg_purity']:.1f}%")
    
    ax.set_xlabel('Cluster Size (Number of Questions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Purity Percentage', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Purity vs Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'purity_vs_size.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: purity_vs_size.png")
    plt.close()


def plot_category_distribution(results: Dict, output_dir: str, top_n: int = 10):
    """Plot category distribution for top clusters."""
    clusters = sorted(results['cluster_details'], key=lambda x: x['total_questions'], reverse=True)[:top_n]
    
    n_cols = 3
    n_rows = (top_n + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if top_n > 1 else [axes]
    
    for idx, cluster in enumerate(clusters):
        ax = axes[idx]
        
        cat_dist = sorted(cluster['category_distribution'].items(), 
                         key=lambda x: x[1]['count'], reverse=True)[:5]
        
        categories = [item[0] for item in cat_dist]
        counts = [item[1]['count'] for item in cat_dist]
        percentages = [item[1]['percentage'] for item in cat_dist]
        
        y_pos = np.arange(len(categories))
        colors_list = ['green' if i == 0 else 'lightblue' for i in range(len(categories))]
        bars = ax.barh(y_pos, counts, color=colors_list, alpha=0.7, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories, fontsize=9)
        ax.set_xlabel('Count', fontsize=9)
        ax.set_title(f'{cluster["cluster_label"][:25]}\n(Purity: {cluster["purity_percentage"]:.1f}%)', 
                    fontsize=10, fontweight='bold')
        ax.invert_yaxis()
        
        for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
            ax.text(count + max(counts)*0.02, i, f'{pct:.1f}%', 
                   va='center', fontsize=8)
    
    for idx in range(len(clusters), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Category Distribution in Top {top_n} Largest Clusters', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'category_distribution_top_clusters.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: category_distribution_top_clusters.png")
    plt.close()


def plot_summary_dashboard(results: Dict, output_dir: str):
    """Create a summary dashboard with key metrics."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    stats = results['overall_stats']
    clusters = results['cluster_details']
    
    # 1. Overall statistics (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    summary_text = f"""
    OVERALL STATISTICS
    
    Total Clusters: {results['total_clusters']}
    Total Questions: {results['total_questions']:,}
    Average Purity: {stats['avg_purity']:.2f}%
    
    High Purity (≥80%): {stats['highly_pure_clusters']}
    Moderate (60-80%): {stats['moderately_pure_clusters']}
    Low Purity (<60%): {stats['mixed_clusters']}
    """
    ax1.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    # 2. Purity levels pie chart
    ax2 = fig.add_subplot(gs[0, 1])
    purity_counts = [stats['highly_pure_clusters'], stats['moderately_pure_clusters'], stats['mixed_clusters']]
    colors_pie = ['green', 'orange', 'red']
    labels_pie = [f'High (≥80%)\n{purity_counts[0]} clusters', 
                  f'Moderate (60-80%)\n{purity_counts[1]} clusters',
                  f'Low (<60%)\n{purity_counts[2]} clusters']
    ax2.pie(purity_counts, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Cluster Purity Distribution', fontsize=12, fontweight='bold')
    
    # 3. Top 10 clusters by size (stacked bar)
    ax3 = fig.add_subplot(gs[0, 2])
    top_10_size = sorted(clusters, key=lambda x: x['total_questions'], reverse=True)[:10]
    cluster_names = [c['cluster_label'][:15] for c in top_10_size]
    sizes = [c['total_questions'] for c in top_10_size]
    x_pos = np.arange(len(cluster_names))
    ax3.bar(x_pos, sizes, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(cluster_names, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Questions')
    ax3.set_title('Top 10 Largest Clusters', fontsize=12, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 4. Purity histogram (middle left)
    ax4 = fig.add_subplot(gs[1, :])
    purity_scores = [c['purity_percentage'] for c in clusters]
    ax4.hist(purity_scores, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(stats['avg_purity'], color='red', linestyle='--', linewidth=2,
               label=f"Average: {stats['avg_purity']:.1f}%")
    ax4.axvline(80, color='green', linestyle='--', alpha=0.5, label='High Purity (80%)')
    ax4.axvline(60, color='orange', linestyle='--', alpha=0.5, label='Moderate Purity (60%)')
    ax4.set_xlabel('Purity Percentage', fontweight='bold')
    ax4.set_ylabel('Number of Clusters', fontweight='bold')
    ax4.set_title('Distribution of Purity Scores Across All Clusters', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Purity vs Size scatter (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    sizes_all = [c['total_questions'] for c in clusters]
    purity_all = [c['purity_percentage'] for c in clusters]
    colors_scatter = ['green' if p >= 80 else 'orange' if p >= 60 else 'red' for p in purity_all]
    ax5.scatter(sizes_all, purity_all, c=colors_scatter, s=100, alpha=0.6, edgecolors='black')
    ax5.axhline(stats['avg_purity'], color='red', linestyle='--', alpha=0.5,
               label=f"Average: {stats['avg_purity']:.1f}%")
    ax5.axhline(80, color='green', linestyle='--', alpha=0.3)
    ax5.axhline(60, color='orange', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Cluster Size (Questions)', fontweight='bold')
    ax5.set_ylabel('Purity Percentage', fontweight='bold')
    ax5.set_title('Cluster Purity vs Size (All Clusters)', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle('Cluster Quality Evaluation Dashboard', fontsize=18, fontweight='bold', y=0.998)
    plt.savefig(Path(output_dir) / 'evaluation_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: evaluation_dashboard.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate intent clustering quality by analyzing category distribution'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the cluster JSON file'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=None,
        help='Show only top N clusters in detailed view (default: all)'
    )
    parser.add_argument(
        '--min-questions',
        type=int,
        default=0,
        help='Minimum questions per cluster to include (default: 0)'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save evaluation results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading cluster data from: {args.input_file}")
    cluster_data = load_clusters(args.input_file)
    
    print(f"Loaded {cluster_data['total_clusters']} clusters")
    
    # Count total questions
    total_questions = sum(len(c.get('questions', [])) for c in cluster_data.get('clusters', []))
    print(f"Total questions: {total_questions:,}")
    
    # Calculate evaluation metrics
    print("\nCalculating category distributions and purity scores...")
    results = calculate_category_distribution(cluster_data)
    
    # Print report
    print_evaluation_report(results, top_n=args.top, min_questions=args.min_questions)
    
    # Save results if requested
    if args.save:
        save_evaluation_results(results, args.save)
    
    # Create plots
    # Generate unique folder name based on input file
    input_name = Path(args.input_file).stem
    plots_dir = f"plots/evaluation_{input_name}"
    create_evaluation_plots(results, plots_dir)
    
    print("\n" + "=" * 100)
    print("Evaluation complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()

