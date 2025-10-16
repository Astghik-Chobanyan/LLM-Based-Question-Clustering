"""
Analyze intent extractions by grouping and showing distributions.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_intent_extractions(file_path: str):
    """Load intent extractions from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def analyze_intent_groups(data, include_all_questions=False, normalize_case=True):
    """Group by extracted_intent and analyze original_intent distribution.
    
    Args:
        data: List of intent extraction items
        include_all_questions: If True, store all questions (can be memory intensive)
        normalize_case: If True, normalize intents to lowercase for grouping
    
    Returns:
        Dictionary of grouped intents
    """
    
    # Group by extracted_intent
    groups = defaultdict(lambda: {
        'count': 0,
        'original_intents': defaultdict(int),
        'categories': defaultdict(int),
        'examples': [],
        'all_questions': [] if include_all_questions else None,
        'original_forms': defaultdict(int) if normalize_case else None
    })
    
    for item in data:
        extracted_intent_original = item['extracted_intent']
        # Normalize to lowercase if requested
        extracted_intent = extracted_intent_original.lower() if normalize_case else extracted_intent_original
        
        original_intent = item['original_intent']
        category = item['category']
        question = item['question']
        
        groups[extracted_intent]['count'] += 1
        groups[extracted_intent]['original_intents'][original_intent] += 1
        groups[extracted_intent]['categories'][category] += 1
        
        # Track original case variations
        if normalize_case:
            groups[extracted_intent]['original_forms'][extracted_intent_original] += 1
        
        # Store up to 3 examples per group for display
        if len(groups[extracted_intent]['examples']) < 3:
            groups[extracted_intent]['examples'].append(question)
        
        # Store all questions if requested
        if include_all_questions:
            groups[extracted_intent]['all_questions'].append({
                'question': question,
                'original_intent': original_intent,
                'category': category,
                'flags': item.get('flags', ''),
                'extracted_intent_original': extracted_intent_original
            })
    
    return groups


def print_analysis(groups, top_n=None, min_count=1):
    """Print the analysis results.
    
    Args:
        groups: Dictionary of grouped intents
        top_n: Show only top N groups by size (None = show all)
        min_count: Minimum questions per group to display
    """
    
    # Sort groups by count
    sorted_groups = sorted(groups.items(), key=lambda x: x[1]['count'], reverse=True)
    
    # Filter by minimum count
    sorted_groups = [(k, v) for k, v in sorted_groups if v['count'] >= min_count]
    
    # Limit to top N if specified
    if top_n:
        sorted_groups = sorted_groups[:top_n]
    
    print("\n" + "=" * 80)
    print("INTENT GROUPS ANALYSIS")
    print("=" * 80)
    
    print(f"\nTotal unique extracted intents: {len(groups)}")
    print(f"Showing {len(sorted_groups)} groups (min {min_count} questions per group)")
    
    for idx, (extracted_intent, info) in enumerate(sorted_groups, 1):
        print("\n" + "-" * 80)
        print(f"\n{idx}. Extracted Intent: '{extracted_intent}'")
        print(f"   Total Questions: {info['count']}")
        
        # Show original case variations if normalized
        if info.get('original_forms'):
            sorted_forms = sorted(info['original_forms'].items(), key=lambda x: x[1], reverse=True)
            if len(sorted_forms) > 1:
                print(f"\n   Case Variations (merged):")
                for form, count in sorted_forms[:5]:
                    print(f"      - '{form}': {count}")
                if len(sorted_forms) > 5:
                    print(f"      - ... and {len(sorted_forms) - 5} more variations")
        
        # Original intent distribution
        print(f"\n   Original Intent Distribution:")
        sorted_original = sorted(info['original_intents'].items(), key=lambda x: x[1], reverse=True)
        for orig_intent, count in sorted_original[:10]:  # Show top 10
            percentage = (count / info['count']) * 100
            print(f"      - {orig_intent}: {count} ({percentage:.1f}%)")
        
        if len(sorted_original) > 10:
            remaining = sum(count for _, count in sorted_original[10:])
            print(f"      - ... and {len(sorted_original) - 10} more ({remaining} questions)")
        
        # Category distribution
        print(f"\n   Category Distribution:")
        sorted_categories = sorted(info['categories'].items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories:
            percentage = (count / info['count']) * 100
            print(f"      - {category}: {count} ({percentage:.1f}%)")
        
        # Example questions
        print(f"\n   Example Questions:")
        for i, example in enumerate(info['examples'][:3], 1):
            print(f"      {i}. {example[:100]}{'...' if len(example) > 100 else ''}")


def save_analysis_to_json(groups, output_path: str, include_all_questions=False):
    """Save analysis results to JSON file.
    
    Args:
        groups: Dictionary of grouped intents
        output_path: Path to save the analysis
        include_all_questions: If True, include all questions in the output
    """
    
    # Convert defaultdicts to regular dicts for JSON serialization
    output_data = {
        'total_groups': len(groups),
        'groups': []
    }
    
    for intent, info in sorted(groups.items(), key=lambda x: x[1]['count'], reverse=True):
        group_data = {
            'extracted_intent': intent,
            'count': info['count'],
            'original_intents': dict(info['original_intents']),
            'categories': dict(info['categories']),
            'examples': info['examples']
        }
        
        # Include original case forms if normalized
        if info.get('original_forms'):
            group_data['original_forms'] = dict(info['original_forms'])
        
        # Include all questions if available and requested
        if include_all_questions and info['all_questions'] is not None:
            group_data['all_questions'] = info['all_questions']
        
        output_data['groups'].append(group_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Analysis saved to: {output_path}")
    if include_all_questions:
        print(f"  (includes all {sum(g['count'] for g in groups.values())} questions)")


def print_summary_statistics(groups):
    """Print summary statistics."""
    
    total_questions = sum(g['count'] for g in groups.values())
    group_sizes = [g['count'] for g in groups.values()]
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\nTotal Questions: {total_questions}")
    print(f"Total Unique Extracted Intents: {len(groups)}")
    print(f"Average Questions per Intent: {total_questions / len(groups):.1f}")
    print(f"Largest Group: {max(group_sizes)}")
    print(f"Smallest Group: {min(group_sizes)}")
    print(f"Median Group Size: {sorted(group_sizes)[len(group_sizes)//2]}")
    
    # Distribution of group sizes
    print(f"\nGroup Size Distribution:")
    size_ranges = [(1, 1), (2, 5), (6, 10), (11, 50), (51, 100), (101, 500), (501, float('inf'))]
    for min_size, max_size in size_ranges:
        count = sum(1 for size in group_sizes if min_size <= size <= max_size)
        if count > 0:
            if max_size == float('inf'):
                print(f"  {min_size}+ questions: {count} groups")
            else:
                print(f"  {min_size}-{max_size} questions: {count} groups")


def main():
    parser = argparse.ArgumentParser(description='Analyze intent extraction groups')
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the intent extractions JSON file'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=None,
        help='Show only top N groups (default: show all)'
    )
    parser.add_argument(
        '--min-count',
        type=int,
        default=1,
        help='Minimum questions per group to display (default: 1)'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save analysis to JSON file (optional)'
    )
    parser.add_argument(
        '--include-all-questions',
        action='store_true',
        help='Include all questions in saved JSON (not just examples)'
    )
    parser.add_argument(
        '--no-normalize-case',
        action='store_true',
        help='Do not normalize intents to lowercase (keep original case distinctions)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.input_file}")
    data = load_intent_extractions(args.input_file)
    print(f"Loaded {len(data)} intent extractions")
    
    # Analyze
    print("\nAnalyzing intent groups...")
    normalize_case = not args.no_normalize_case
    if normalize_case:
        print("(normalizing to lowercase - merging case variations)")
    else:
        print("(preserving original case - keeping case distinctions)")
    if args.include_all_questions:
        print("(including all questions in groups - may use more memory)")
    groups = analyze_intent_groups(data, 
                                   include_all_questions=args.include_all_questions,
                                   normalize_case=normalize_case)
    
    # Print summary statistics
    print_summary_statistics(groups)
    
    # Print detailed analysis
    print_analysis(groups, top_n=args.top, min_count=args.min_count)
    
    # Save to JSON if requested
    if args.save:
        save_analysis_to_json(groups, args.save, include_all_questions=args.include_all_questions)
    
    # Print top 100 intents by question count
    print("\n" + "=" * 80)
    print("TOP 100 INTENTS BY QUESTION COUNT")
    print("=" * 80)
    
    sorted_by_count = sorted(groups.items(), key=lambda x: x[1]['count'], reverse=True)[:100]
    
    print(f"\n{'Rank':<6} {'Questions':<12} {'Extracted Intent'}")
    print("-" * 80)
    
    for rank, (intent, info) in enumerate(sorted_by_count, 1):
        count = info['count']
        print(f"{rank:<6} {count:<12} {intent}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

