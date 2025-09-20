#!/usr/bin/env python3
"""
Comprehensive Example Usage of HMIC Algorithm

This script demonstrates various use cases and scenarios for the HMIC algorithm:
1. Basic HMIC implementation on Facebook data
2. Synthetic network testing
3. Parameter optimization
4. Comparison with baseline algorithms
5. Statistical analysis
6. Visualization and reporting
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import Dict, List

from hmic_algorithm import HMICAlgorithm
from hmic_analysis import HMICAnalysis


def example_1_basic_hmic():
    """
    Example 1: Basic HMIC implementation on Facebook data
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic HMIC Implementation")
    print("="*60)
    
    # Initialize algorithm
    hmic = HMICAlgorithm(random_seed=42)
    
    # Load Facebook data
    hmic.load_facebook_data("facebook_combined.txt")
    
    # Run complete experiment
    results = hmic.run_complete_experiment(budget=15, iterations=3)
    
    if results:
        # Visualize results
        hmic.visualize_results(results, save_path="example1_basic_results.png")
        
        # Compare with baseline
        comparison = hmic.compare_with_baseline(results)
        
        # Save results
        hmic.save_results(results, "example1_basic_results.json")
        
        print(f"\nBasic HMIC Results:")
        print(f"  Mastermind: Node {results['mastermind']}")
        print(f"  Community size: {results['parameters']['community_size']}")
        print(f"  Intermediate nodes: {results['parameters']['intermediate_nodes_count']}")
        print(f"  Execution time: {results['execution_time']:.2f} seconds")
        
        return results
    else:
        print("Basic HMIC experiment failed.")
        return None


def example_2_synthetic_network():
    """
    Example 2: HMIC on synthetic network with planted communities
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Synthetic Network Testing")
    print("="*60)
    
    # Initialize algorithm
    hmic = HMICAlgorithm(random_seed=42)
    
    # Create synthetic network
    hmic.create_synthetic_network(n_nodes=500, p_in=0.15, p_out=0.02)
    
    # Run experiment
    results = hmic.run_complete_experiment(budget=10, iterations=2)
    
    if results:
        # Visualize results
        hmic.visualize_results(results, save_path="example2_synthetic_results.png")
        
        # Save results
        hmic.save_results(results, "example2_synthetic_results.json")
        
        print(f"\nSynthetic Network Results:")
        print(f"  Mastermind: Node {results['mastermind']}")
        print(f"  Community size: {results['parameters']['community_size']}")
        print(f"  Intermediate nodes: {results['parameters']['intermediate_nodes_count']}")
        
        return results
    else:
        print("Synthetic network experiment failed.")
        return None


def example_3_parameter_optimization():
    """
    Example 3: Parameter optimization for HMIC
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Parameter Optimization")
    print("="*60)
    
    # Initialize analysis
    analysis = HMICAnalysis(random_seed=42)
    
    # Load data
    hmic = HMICAlgorithm(random_seed=42)
    hmic.load_facebook_data("facebook_combined.txt")
    
    # Run basic experiment to get mastermind
    basic_results = hmic.run_complete_experiment(budget=10, iterations=2)
    
    if basic_results:
        # Parameter optimization
        optimization = analysis.parameter_optimization(
            hmic.original_graph,
            basic_results['mastermind'],
            budget_range=[5, 10, 15, 20],
            iteration_range=[1, 2, 3]
        )
        
        # Visualize optimization
        analysis.visualize_parameter_optimization(optimization, save_path="example3_optimization.png")
        
        # Print optimal parameters
        print("\nOptimal Parameters:")
        print(f"  Best centrality: Budget={optimization['optimal_parameters']['best_centrality']['parameters'][0]}, "
              f"Iterations={optimization['optimal_parameters']['best_centrality']['parameters'][1]}")
        print(f"  Best SIR: Budget={optimization['optimal_parameters']['best_sir']['parameters'][0]}, "
              f"Iterations={optimization['optimal_parameters']['best_sir']['parameters'][1]}")
        print(f"  Best balanced: Budget={optimization['optimal_parameters']['best_balanced']['parameters'][0]}, "
              f"Iterations={optimization['optimal_parameters']['best_balanced']['parameters'][1]}")
        
        # Save optimization results
        with open("example3_optimization_results.json", 'w') as f:
            json.dump(optimization, f, indent=2)
        
        return optimization
    else:
        print("Parameter optimization failed.")
        return None


def example_4_algorithm_comparison():
    """
    Example 4: Compare HMIC with ROAM algorithm
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Algorithm Comparison")
    print("="*60)
    
    # Initialize analysis
    analysis = HMICAnalysis(random_seed=42)
    
    # Load data
    hmic = HMICAlgorithm(random_seed=42)
    hmic.load_facebook_data("facebook_combined.txt")
    
    # Run basic experiment to get mastermind
    basic_results = hmic.run_complete_experiment(budget=15, iterations=3)
    
    if basic_results:
        # Compare with ROAM
        comparison = analysis.compare_with_roam_algorithm(
            hmic.original_graph,
            basic_results['mastermind'],
            budget=15,
            iterations=3
        )
        
        # Visualize comparison
        analysis.visualize_comparison(comparison, save_path="example4_comparison.png")
        
        # Print comparison results
        print("\nAlgorithm Comparison Results:")
        print(f"  HMIC centrality reduction: {comparison['hmic_results']['centrality_reduction']:.4f}")
        print(f"  ROAM centrality reduction: {comparison['roam_results']['centrality_reduction']:.4f}")
        print(f"  HMIC improvement: {comparison['effectiveness_comparison']['centrality_comparison']['improvement']:.4f}")
        
        print(f"  HMIC SIR reduction: {comparison['hmic_results']['sir_reduction']:.2f}%")
        print(f"  ROAM SIR reduction: {comparison['roam_results']['sir_reduction']:.2f}%")
        print(f"  HMIC improvement: {comparison['effectiveness_comparison']['sir_comparison']['improvement']:.2f}%")
        
        # Save comparison results
        with open("example4_comparison_results.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison
    else:
        print("Algorithm comparison failed.")
        return None


def example_5_statistical_analysis():
    """
    Example 5: Statistical analysis of HMIC performance
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Statistical Analysis")
    print("="*60)
    
    # Initialize analysis
    analysis = HMICAnalysis(random_seed=42)
    
    # Load data
    hmic = HMICAlgorithm(random_seed=42)
    hmic.load_facebook_data("facebook_combined.txt")
    
    # Run basic experiment
    basic_results = hmic.run_complete_experiment(budget=15, iterations=3)
    
    if basic_results:
        # Add original graph to results for statistical analysis
        basic_results['original_graph'] = hmic.original_graph
        
        # Statistical analysis
        stats = analysis.statistical_analysis(basic_results, num_trials=5)
        
        # Print statistical results
        print("\nStatistical Analysis Results:")
        print(f"  Centrality reduction: {stats['centrality_reduction']['mean']:.4f} ± {stats['centrality_reduction']['std']:.4f}")
        print(f"  SIR reduction: {stats['sir_reduction']['mean']:.2f}% ± {stats['sir_reduction']['std']:.2f}%")
        print(f"  Correlation: {stats['correlation']:.4f}")
        
        print(f"  Centrality 95% CI: [{stats['centrality_reduction']['confidence_interval_95'][0]:.4f}, "
              f"{stats['centrality_reduction']['confidence_interval_95'][1]:.4f}]")
        print(f"  SIR 95% CI: [{stats['sir_reduction']['confidence_interval_95'][0]:.2f}%, "
              f"{stats['sir_reduction']['confidence_interval_95'][1]:.2f}%]")
        
        # Save statistical results
        with open("example5_statistical_results.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    else:
        print("Statistical analysis failed.")
        return None


def example_6_multiple_communities():
    """
    Example 6: Test HMIC on multiple communities
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Multiple Communities Analysis")
    print("="*60)
    
    # Initialize algorithm
    hmic = HMICAlgorithm(random_seed=42)
    
    # Load Facebook data
    hmic.load_facebook_data("facebook_combined.txt")
    
    # Detect communities
    communities = hmic.detect_communities_greedy_modularity()
    
    # Test on first 3 communities
    results = {}
    
    for i in range(min(3, len(communities))):
        print(f"\n--- Testing Community {i} ---")
        
        # Run experiment for this community
        community_results = hmic.run_complete_experiment(budget=10, iterations=2, community_index=i)
        
        if community_results:
            results[f"community_{i}"] = community_results
            
            print(f"  Mastermind: Node {community_results['mastermind']}")
            print(f"  Community size: {community_results['parameters']['community_size']}")
            print(f"  Intermediate nodes: {community_results['parameters']['intermediate_nodes_count']}")
    
    # Save multi-community results
    if results:
        with open("example6_multi_community_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nMulti-community analysis completed for {len(results)} communities.")
        return results
    else:
        print("Multi-community analysis failed.")
        return None


def example_7_performance_benchmark():
    """
    Example 7: Performance benchmarking
    """
    print("\n" + "="*60)
    print("EXAMPLE 7: Performance Benchmarking")
    print("="*60)
    
    # Test different network sizes
    network_sizes = [100, 500, 1000]
    results = {}
    
    for size in network_sizes:
        print(f"\n--- Testing Network Size {size} ---")
        
        # Create synthetic network
        hmic = HMICAlgorithm(random_seed=42)
        hmic.create_synthetic_network(n_nodes=size, p_in=0.1, p_out=0.01)
        
        # Measure execution time
        start_time = time.time()
        experiment_results = hmic.run_complete_experiment(budget=10, iterations=2)
        execution_time = time.time() - start_time
        
        if experiment_results:
            results[f"size_{size}"] = {
                'nodes': hmic.graph.number_of_nodes(),
                'edges': hmic.graph.number_of_edges(),
                'execution_time': execution_time,
                'mastermind': experiment_results['mastermind'],
                'community_size': experiment_results['parameters']['community_size'],
                'intermediate_nodes': experiment_results['parameters']['intermediate_nodes_count']
            }
            
            print(f"  Nodes: {hmic.graph.number_of_nodes()}")
            print(f"  Edges: {hmic.graph.number_of_edges()}")
            print(f"  Execution time: {execution_time:.2f} seconds")
    
    # Save benchmark results
    if results:
        with open("example7_benchmark_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nPerformance benchmarking completed for {len(results)} network sizes.")
        return results
    else:
        print("Performance benchmarking failed.")
        return None


def generate_comprehensive_report():
    """
    Generate a comprehensive report of all examples
    """
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*60)
    
    # Run all examples
    all_results = {}
    
    # Example 1: Basic HMIC
    all_results['basic_hmic'] = example_1_basic_hmic()
    
    # Example 2: Synthetic network
    all_results['synthetic_network'] = example_2_synthetic_network()
    
    # Example 3: Parameter optimization
    all_results['parameter_optimization'] = example_3_parameter_optimization()
    
    # Example 4: Algorithm comparison
    all_results['algorithm_comparison'] = example_4_algorithm_comparison()
    
    # Example 5: Statistical analysis
    all_results['statistical_analysis'] = example_5_statistical_analysis()
    
    # Example 6: Multiple communities
    all_results['multiple_communities'] = example_6_multiple_communities()
    
    # Example 7: Performance benchmark
    all_results['performance_benchmark'] = example_7_performance_benchmark()
    
    # Generate summary report
    print("\n" + "="*60)
    print("COMPREHENSIVE REPORT SUMMARY")
    print("="*60)
    
    successful_examples = 0
    for example_name, result in all_results.items():
        if result is not None:
            successful_examples += 1
            print(f"✓ {example_name}: SUCCESS")
        else:
            print(f"✗ {example_name}: FAILED")
    
    print(f"\nOverall Success Rate: {successful_examples}/{len(all_results)} ({successful_examples/len(all_results)*100:.1f}%)")
    
    # Save comprehensive results
    with open("comprehensive_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nComprehensive results saved to 'comprehensive_results.json'")
    
    return all_results


def main():
    """
    Main function to run all examples
    """
    print("HMIC Algorithm - Comprehensive Example Usage")
    print("="*60)
    print("This script demonstrates various use cases of the HMIC algorithm.")
    print("Running all examples may take several minutes...")
    
    # Generate comprehensive report
    results = generate_comprehensive_report()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED")
    print("="*60)
    print("Check the generated files for detailed results:")
    print("  - example1_basic_results.png/json")
    print("  - example2_synthetic_results.png/json")
    print("  - example3_optimization.png/json")
    print("  - example4_comparison.png/json")
    print("  - example5_statistical_results.json")
    print("  - example6_multi_community_results.json")
    print("  - example7_benchmark_results.json")
    print("  - comprehensive_results.json")


if __name__ == "__main__":
    main()

