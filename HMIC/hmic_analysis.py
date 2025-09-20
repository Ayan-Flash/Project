import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import time
from scipy.stats import pearsonr, ttest_ind
from sklearn.metrics import normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')

from hmic_algorithm import HMICAlgorithm


class HMICAnalysis:
    """
    Advanced analysis and comparison tools for HMIC algorithm
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize HMIC Analysis
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    def compare_with_roam_algorithm(self, graph: nx.Graph, mastermind: int, 
                                  budget: int, iterations: int) -> Dict:
        """
        Compare HMIC with ROAM (Remove and Add Method) algorithm
        
        Args:
            graph: Network graph
            mastermind: Mastermind node
            budget: Budget for edge modifications
            iterations: Number of iterations
            
        Returns:
            Comparison results
        """
        print("Comparing HMIC with ROAM algorithm...")
        
        # Initialize both algorithms
        hmic = HMICAlgorithm(self.random_seed)
        hmic.graph = graph.copy()
        
        # Run HMIC
        hmic_results = self._run_hmic_variant(hmic, mastermind, budget, iterations)
        
        # Run ROAM
        roam_results = self._run_roam_algorithm(graph, mastermind, budget, iterations)
        
        # Compare results
        comparison = {
            'hmic_results': hmic_results,
            'roam_results': roam_results,
            'effectiveness_comparison': self._compare_effectiveness(hmic_results, roam_results)
        }
        
        return comparison
    
    def _run_hmic_variant(self, hmic: HMICAlgorithm, mastermind: int, 
                         budget: int, iterations: int) -> Dict:
        """
        Run HMIC variant for comparison
        """
        # Detect communities
        communities = hmic.detect_communities_greedy_modularity()
        
        # Find mastermind's community
        mastermind_community = None
        for comm in communities:
            if mastermind in comm:
                mastermind_community = comm
                break
        
        if not mastermind_community:
            return {}
        
        # Identify intermediate nodes
        intermediate_nodes = hmic.identify_intermediate_nodes(mastermind_community, mastermind)
        
        # Apply HMIC strategy
        results = hmic.apply_hmic_strategy(budget, iterations)
        
        return {
            'centrality_reduction': results[iterations-1]['centrality_reduction']['average'] if results else 0,
            'sir_reduction': self._calculate_sir_reduction(hmic, mastermind, results),
            'iterations': iterations
        }
    
    def _run_roam_algorithm(self, graph: nx.Graph, mastermind: int, 
                           budget: int, iterations: int) -> Dict:
        """
        Implement ROAM (Remove and Add Method) algorithm
        
        ROAM randomly removes edges from mastermind and adds edges to random nodes
        """
        print("Running ROAM algorithm...")
        
        graph_copy = graph.copy()
        original_centrality = self._calculate_centrality(graph_copy, mastermind)
        
        for iteration in range(iterations):
            # Get mastermind's current edges
            mastermind_edges = list(graph_copy.edges(mastermind))
            
            if not mastermind_edges:
                break
            
            # Randomly remove edges from mastermind
            edges_to_remove = min(budget, len(mastermind_edges))
            edges_removed = random.sample(mastermind_edges, edges_to_remove)
            
            for edge in edges_removed:
                graph_copy.remove_edge(*edge)
            
            # Add edges to random nodes
            nodes = list(graph_copy.nodes())
            nodes.remove(mastermind)
            
            for _ in range(budget):
                if len(nodes) >= 2:
                    node1, node2 = random.sample(nodes, 2)
                    if not graph_copy.has_edge(node1, node2):
                        graph_copy.add_edge(node1, node2)
        
        # Calculate final centrality
        final_centrality = self._calculate_centrality(graph_copy, mastermind)
        
        return {
            'centrality_reduction': original_centrality - final_centrality,
            'sir_reduction': self._calculate_sir_reduction_simple(graph_copy, mastermind),
            'iterations': iterations
        }
    
    def _calculate_centrality(self, graph: nx.Graph, node: int) -> float:
        """
        Calculate average centrality for a node
        """
        try:
            degree_cent = nx.degree_centrality(graph)[node]
            closeness_cent = nx.closeness_centrality(graph)[node]
            betweenness_cent = nx.betweenness_centrality(graph)[node]
            return (degree_cent + closeness_cent + betweenness_cent) / 3
        except:
            return 0
    
    def _calculate_sir_reduction(self, hmic: HMICAlgorithm, mastermind: int, results: Dict) -> float:
        """
        Calculate SIR reduction for HMIC
        """
        if not results:
            return 0
        
        # Get original and final SIR results
        original_sir = results[0]['sir_results']['total_infected']
        final_sir = results[len(results)-1]['sir_results']['total_infected']
        
        if original_sir == 0:
            return 0
        
        return (original_sir - final_sir) / original_sir * 100
    
    def _calculate_sir_reduction_simple(self, graph: nx.Graph, mastermind: int) -> float:
        """
        Calculate SIR reduction for simple comparison
        """
        # Simple SIR simulation
        infected = {mastermind}
        susceptible = set(graph.nodes()) - infected
        
        for _ in range(10):  # Simplified simulation
            new_infected = set()
            for inf_node in infected:
                for neighbor in graph.neighbors(inf_node):
                    if neighbor in susceptible and random.random() < 0.3:
                        new_infected.add(neighbor)
                        susceptible.remove(neighbor)
            infected.update(new_infected)
        
        return len(infected)
    
    def _compare_effectiveness(self, hmic_results: Dict, roam_results: Dict) -> Dict:
        """
        Compare effectiveness of HMIC vs ROAM
        """
        return {
            'centrality_comparison': {
                'hmic': hmic_results['centrality_reduction'],
                'roam': roam_results['centrality_reduction'],
                'improvement': hmic_results['centrality_reduction'] - roam_results['centrality_reduction']
            },
            'sir_comparison': {
                'hmic': hmic_results['sir_reduction'],
                'roam': roam_results['sir_reduction'],
                'improvement': hmic_results['sir_reduction'] - roam_results['sir_reduction']
            }
        }
    
    def parameter_optimization(self, graph: nx.Graph, mastermind: int, 
                             budget_range: List[int], iteration_range: List[int]) -> Dict:
        """
        Optimize HMIC parameters (budget and iterations)
        
        Args:
            graph: Network graph
            mastermind: Mastermind node
            budget_range: Range of budgets to test
            iteration_range: Range of iterations to test
            
        Returns:
            Optimization results
        """
        print("Running parameter optimization...")
        
        results = {}
        
        for budget in budget_range:
            for iterations in iteration_range:
                print(f"Testing budget={budget}, iterations={iterations}")
                
                hmic = HMICAlgorithm(self.random_seed)
                hmic.graph = graph.copy()
                
                # Run experiment
                experiment_results = self._run_parameter_test(hmic, mastermind, budget, iterations)
                
                results[(budget, iterations)] = experiment_results
        
        # Find optimal parameters
        optimal_params = self._find_optimal_parameters(results)
        
        return {
            'detailed_results': results,
            'optimal_parameters': optimal_params
        }
    
    def _run_parameter_test(self, hmic: HMICAlgorithm, mastermind: int, 
                           budget: int, iterations: int) -> Dict:
        """
        Run single parameter test
        """
        # Detect communities
        communities = hmic.detect_communities_greedy_modularity()
        
        # Find mastermind's community
        mastermind_community = None
        for comm in communities:
            if mastermind in comm:
                mastermind_community = comm
                break
        
        if not mastermind_community:
            return {'centrality_reduction': 0, 'sir_reduction': 0}
        
        # Identify intermediate nodes
        intermediate_nodes = hmic.identify_intermediate_nodes(mastermind_community, mastermind)
        
        # Apply HMIC strategy
        results = hmic.apply_hmic_strategy(budget, iterations)
        
        if not results:
            return {'centrality_reduction': 0, 'sir_reduction': 0}
        
        centrality_reduction = results[iterations-1]['centrality_reduction']['average']
        sir_reduction = self._calculate_sir_reduction(hmic, mastermind, results)
        
        return {
            'centrality_reduction': centrality_reduction,
            'sir_reduction': sir_reduction
        }
    
    def _find_optimal_parameters(self, results: Dict) -> Dict:
        """
        Find optimal parameters based on results
        """
        # Find best centrality reduction
        best_centrality = max(results.items(), 
                            key=lambda x: x[1]['centrality_reduction'])
        
        # Find best SIR reduction
        best_sir = max(results.items(), 
                      key=lambda x: x[1]['sir_reduction'])
        
        # Find balanced solution (weighted combination)
        best_balanced = max(results.items(), 
                           key=lambda x: x[1]['centrality_reduction'] * 0.6 + x[1]['sir_reduction'] * 0.4)
        
        return {
            'best_centrality': {
                'parameters': best_centrality[0],
                'results': best_centrality[1]
            },
            'best_sir': {
                'parameters': best_sir[0],
                'results': best_sir[1]
            },
            'best_balanced': {
                'parameters': best_balanced[0],
                'results': best_balanced[1]
            }
        }
    
    def statistical_analysis(self, hmic_results: Dict, num_trials: int = 10) -> Dict:
        """
        Perform statistical analysis on HMIC results
        
        Args:
            hmic_results: HMIC experiment results
            num_trials: Number of trials for statistical analysis
            
        Returns:
            Statistical analysis results
        """
        print(f"Performing statistical analysis with {num_trials} trials...")
        
        centrality_reductions = []
        sir_reductions = []
        
        # Run multiple trials
        for trial in range(num_trials):
            # Create new instance with different seed
            hmic = HMICAlgorithm(self.random_seed + trial)
            hmic.graph = hmic_results['original_graph'].copy()
            
            # Run experiment
            trial_results = self._run_statistical_trial(hmic, hmic_results['mastermind'], 
                                                      hmic_results['parameters']['budget'],
                                                      hmic_results['parameters']['iterations'])
            
            centrality_reductions.append(trial_results['centrality_reduction'])
            sir_reductions.append(trial_results['sir_reduction'])
        
        # Calculate statistics
        stats = {
            'centrality_reduction': {
                'mean': np.mean(centrality_reductions),
                'std': np.std(centrality_reductions),
                'min': np.min(centrality_reductions),
                'max': np.max(centrality_reductions),
                'confidence_interval_95': self._calculate_confidence_interval(centrality_reductions, 0.95)
            },
            'sir_reduction': {
                'mean': np.mean(sir_reductions),
                'std': np.std(sir_reductions),
                'min': np.min(sir_reductions),
                'max': np.max(sir_reductions),
                'confidence_interval_95': self._calculate_confidence_interval(sir_reductions, 0.95)
            },
            'correlation': pearsonr(centrality_reductions, sir_reductions)[0]
        }
        
        return stats
    
    def _run_statistical_trial(self, hmic: HMICAlgorithm, mastermind: int, 
                              budget: int, iterations: int) -> Dict:
        """
        Run single statistical trial
        """
        # Detect communities
        communities = hmic.detect_communities_greedy_modularity()
        
        # Find mastermind's community
        mastermind_community = None
        for comm in communities:
            if mastermind in comm:
                mastermind_community = comm
                break
        
        if not mastermind_community:
            return {'centrality_reduction': 0, 'sir_reduction': 0}
        
        # Identify intermediate nodes
        intermediate_nodes = hmic.identify_intermediate_nodes(mastermind_community, mastermind)
        
        # Apply HMIC strategy
        results = hmic.apply_hmic_strategy(budget, iterations)
        
        if not results:
            return {'centrality_reduction': 0, 'sir_reduction': 0}
        
        centrality_reduction = results[iterations-1]['centrality_reduction']['average']
        sir_reduction = self._calculate_sir_reduction(hmic, mastermind, results)
        
        return {
            'centrality_reduction': centrality_reduction,
            'sir_reduction': sir_reduction
        }
    
    def _calculate_confidence_interval(self, data: List[float], confidence: float) -> Tuple[float, float]:
        """
        Calculate confidence interval
        """
        mean = np.mean(data)
        std = np.std(data)
        n = len(data)
        
        # Z-score for confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)
        
        margin_of_error = z * (std / np.sqrt(n))
        
        return (mean - margin_of_error, mean + margin_of_error)
    
    def visualize_comparison(self, comparison_results: Dict, save_path: str = None) -> None:
        """
        Visualize comparison between HMIC and ROAM
        
        Args:
            comparison_results: Comparison results
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('HMIC vs ROAM Algorithm Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Centrality reduction comparison
        algorithms = ['HMIC', 'ROAM']
        centrality_reductions = [
            comparison_results['hmic_results']['centrality_reduction'],
            comparison_results['roam_results']['centrality_reduction']
        ]
        
        bars1 = axes[0].bar(algorithms, centrality_reductions, color=['blue', 'orange'], alpha=0.7)
        axes[0].set_title('Centrality Reduction Comparison')
        axes[0].set_ylabel('Centrality Reduction')
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, centrality_reductions):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.4f}', ha='center', va='bottom')
        
        # Plot 2: SIR reduction comparison
        sir_reductions = [
            comparison_results['hmic_results']['sir_reduction'],
            comparison_results['roam_results']['sir_reduction']
        ]
        
        bars2 = axes[1].bar(algorithms, sir_reductions, color=['green', 'red'], alpha=0.7)
        axes[1].set_title('SIR Reduction Comparison')
        axes[1].set_ylabel('SIR Reduction (%)')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, sir_reductions):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_parameter_optimization(self, optimization_results: Dict, save_path: str = None) -> None:
        """
        Visualize parameter optimization results
        
        Args:
            optimization_results: Parameter optimization results
            save_path: Path to save visualization
        """
        results = optimization_results['detailed_results']
        
        # Extract parameters and results
        budgets = []
        iterations = []
        centrality_reductions = []
        sir_reductions = []
        
        for (budget, iteration), result in results.items():
            budgets.append(budget)
            iterations.append(iteration)
            centrality_reductions.append(result['centrality_reduction'])
            sir_reductions.append(result['sir_reduction'])
        
        # Create meshgrid for 3D plotting
        unique_budgets = sorted(list(set(budgets)))
        unique_iterations = sorted(list(set(iterations)))
        
        X, Y = np.meshgrid(unique_budgets, unique_iterations)
        
        # Create Z matrices
        Z_centrality = np.zeros((len(unique_iterations), len(unique_budgets)))
        Z_sir = np.zeros((len(unique_iterations), len(unique_budgets)))
        
        for i, iteration in enumerate(unique_iterations):
            for j, budget in enumerate(unique_budgets):
                if (budget, iteration) in results:
                    Z_centrality[i, j] = results[(budget, iteration)]['centrality_reduction']
                    Z_sir[i, j] = results[(budget, iteration)]['sir_reduction']
        
        # Create 3D plots
        fig = plt.figure(figsize=(15, 6))
        
        # Plot 1: Centrality reduction
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z_centrality, cmap='viridis', alpha=0.8)
        ax1.set_title('Centrality Reduction vs Parameters')
        ax1.set_xlabel('Budget')
        ax1.set_ylabel('Iterations')
        ax1.set_zlabel('Centrality Reduction')
        
        # Plot 2: SIR reduction
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z_sir, cmap='plasma', alpha=0.8)
        ax2.set_title('SIR Reduction vs Parameters')
        ax2.set_xlabel('Budget')
        ax2.set_ylabel('Iterations')
        ax2.set_zlabel('SIR Reduction (%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter optimization visualization saved to {save_path}")
        
        plt.show()


def main():
    """
    Main function to demonstrate advanced analysis
    """
    print("HMIC Advanced Analysis Demonstration")
    print("=" * 50)
    
    # Initialize analysis
    analysis = HMICAnalysis(random_seed=42)
    
    # Load data
    hmic = HMICAlgorithm(random_seed=42)
    hmic.load_facebook_data("facebook_combined.txt")
    
    # Run basic HMIC experiment
    results = hmic.run_complete_experiment(budget=15, iterations=3)
    
    if results:
        # Compare with ROAM
        comparison = analysis.compare_with_roam_algorithm(
            hmic.original_graph, 
            results['mastermind'], 
            budget=15, 
            iterations=3
        )
        
        # Visualize comparison
        analysis.visualize_comparison(comparison, save_path="hmic_vs_roam.png")
        
        # Parameter optimization
        optimization = analysis.parameter_optimization(
            hmic.original_graph,
            results['mastermind'],
            budget_range=[5, 10, 15, 20],
            iteration_range=[1, 2, 3, 4]
        )
        
        # Visualize optimization
        analysis.visualize_parameter_optimization(optimization, save_path="parameter_optimization.png")
        
        # Statistical analysis
        stats = analysis.statistical_analysis(results, num_trials=5)
        
        print("\nStatistical Analysis Results:")
        print(f"Centrality reduction: {stats['centrality_reduction']['mean']:.4f} ± {stats['centrality_reduction']['std']:.4f}")
        print(f"SIR reduction: {stats['sir_reduction']['mean']:.2f}% ± {stats['sir_reduction']['std']:.2f}%")
        print(f"Correlation: {stats['correlation']:.4f}")
        
        print("\nAdvanced analysis completed successfully!")


if __name__ == "__main__":
    main()

