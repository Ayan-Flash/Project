import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import time
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class HMICAlgorithm:
    """
    HMIC (Hide Mastermind using Intermediate Connection) Algorithm Implementation
    
    This class implements the complete HMIC algorithm for hiding influential nodes
    in social networks through strategic edge modifications.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize HMIC Algorithm
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        self.graph = None
        self.original_graph = None
        self.communities = None
        self.mastermind = None
        self.intermediate_nodes = None
        self.results = {}
        
    def load_facebook_data(self, file_path: str) -> None:
        """
        Load Facebook social network data
        
        Args:
            file_path: Path to the Facebook combined dataset
        """
        print(f"Loading Facebook data from {file_path}...")
        self.graph = nx.read_edgelist(file_path, nodetype=int)
        self.original_graph = self.graph.copy()
        
        # Remove self-loops and ensure undirected
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
        self.graph = self.graph.to_undirected()
        
        # Keep only largest connected component
        largest_cc = max(nx.connected_components(self.graph), key=len)
        self.graph = self.graph.subgraph(largest_cc).copy()
        
        print(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
    def create_synthetic_network(self, n_nodes: int = 1000, p_in: float = 0.1, p_out: float = 0.01) -> None:
        """
        Create a synthetic network with planted communities for testing
        
        Args:
            n_nodes: Number of nodes
            p_in: Probability of edge within communities
            p_out: Probability of edge between communities
        """
        print(f"Creating synthetic network with {n_nodes} nodes...")
        
        # Create planted partition
        n_communities = 5
        community_size = n_nodes // n_communities
        communities = []
        
        for i in range(n_communities):
            start = i * community_size
            end = start + community_size if i < n_communities - 1 else n_nodes
            communities.append(list(range(start, end)))
        
        # Create graph with planted communities
        self.graph = nx.random_partition_graph(
            [len(comm) for comm in communities], 
            p_in, 
            p_out, 
            seed=self.random_seed
        )
        
        self.original_graph = self.graph.copy()
        print(f"Created synthetic graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
    def detect_communities_greedy_modularity(self) -> List[Set]:
        """
        Detect communities using greedy modularity maximization
        
        Returns:
            List of communities (sets of nodes)
        """
        print("Detecting communities using greedy modularity...")
        
        # Use NetworkX's greedy modularity communities
        communities = list(nx.community.greedy_modularity_communities(self.graph))
        
        # Filter out very small communities (less than 5 nodes)
        communities = [comm for comm in communities if len(comm) >= 5]
        
        print(f"Detected {len(communities)} communities")
        for i, comm in enumerate(communities):
            print(f"  Community {i}: {len(comm)} nodes")
            
        self.communities = communities
        return communities
    
    def calculate_centrality_metrics(self, node: int) -> Dict[str, float]:
        """
        Calculate centrality metrics for a given node
        
        Args:
            node: Node identifier
            
        Returns:
            Dictionary with centrality metrics
        """
        try:
            degree_cent = nx.degree_centrality(self.graph)[node]
            closeness_cent = nx.closeness_centrality(self.graph)[node]
            betweenness_cent = nx.betweenness_centrality(self.graph)[node]
            
            return {
                'degree': degree_cent,
                'closeness': closeness_cent,
                'betweenness': betweenness_cent,
                'average': (degree_cent + closeness_cent + betweenness_cent) / 3
            }
        except:
            return {'degree': 0, 'closeness': 0, 'betweenness': 0, 'average': 0}
    
    def identify_mastermind(self, community: Set[int]) -> int:
        """
        Identify the mastermind node in a community based on centrality metrics
        
        Args:
            community: Set of nodes in the community
            
        Returns:
            Mastermind node identifier
        """
        print(f"Identifying mastermind in community with {len(community)} nodes...")
        
        # Calculate centrality for all nodes in community
        centrality_scores = {}
        for node in community:
            centrality_scores[node] = self.calculate_centrality_metrics(node)
        
        # Find node with highest average centrality
        mastermind = max(centrality_scores.keys(), 
                        key=lambda x: centrality_scores[x]['average'])
        
        print(f"Mastermind identified: Node {mastermind}")
        print(f"  Degree centrality: {centrality_scores[mastermind]['degree']:.4f}")
        print(f"  Closeness centrality: {centrality_scores[mastermind]['closeness']:.4f}")
        print(f"  Betweenness centrality: {centrality_scores[mastermind]['betweenness']:.4f}")
        
        self.mastermind = mastermind
        return mastermind
    
    def identify_intermediate_nodes(self, community: Set[int], mastermind: int) -> Set[int]:
        """
        Identify intermediate nodes that connect the mastermind's community to external nodes
        
        Args:
            community: Set of nodes in the community
            mastermind: Mastermind node identifier
            
        Returns:
            Set of intermediate node identifiers
        """
        print("Identifying intermediate nodes...")
        
        # Find nodes in the community that have connections outside the community
        intermediate_nodes = set()
        
        for node in community:
            if node == mastermind:
                continue
                
            # Get neighbors of this node
            neighbors = set(self.graph.neighbors(node))
            
            # Check if node has connections outside the community
            external_connections = neighbors - community
            
            if external_connections:
                intermediate_nodes.add(node)
        
        print(f"Found {len(intermediate_nodes)} intermediate nodes")
        self.intermediate_nodes = intermediate_nodes
        return intermediate_nodes
    
    def simulate_sir_epidemic(self, initial_infected: List[int], 
                            infection_rate: float = 0.3, 
                            recovery_rate: float = 0.1, 
                            max_steps: int = 50) -> Dict:
        """
        Simulate SIR epidemic spread from initial infected nodes
        
        Args:
            initial_infected: List of initially infected nodes
            infection_rate: Probability of infection per contact
            recovery_rate: Probability of recovery per step
            max_steps: Maximum simulation steps
            
        Returns:
            Dictionary with simulation results
        """
        if not self.graph:
            return {'total_infected': 0, 'peak_infected': 0, 'steps': 0}
        
        # Initialize states
        susceptible = set(self.graph.nodes()) - set(initial_infected)
        infected = set(initial_infected)
        recovered = set()
        
        infected_history = [len(infected)]
        
        for step in range(max_steps):
            new_infected = set()
            new_recovered = set()
            
            # Process infections
            for inf_node in infected:
                for neighbor in self.graph.neighbors(inf_node):
                    if neighbor in susceptible:
                        if random.random() < infection_rate:
                            new_infected.add(neighbor)
                            susceptible.remove(neighbor)
            
            # Process recoveries
            for inf_node in infected:
                if random.random() < recovery_rate:
                    new_recovered.add(inf_node)
            
            # Update states
            infected.update(new_infected)
            infected -= new_recovered
            recovered.update(new_recovered)
            
            infected_history.append(len(infected))
            
            # Stop if no more infected nodes
            if len(infected) == 0:
                break
        
        return {
            'total_infected': len(recovered) + len(infected),
            'peak_infected': max(infected_history),
            'steps': len(infected_history),
            'history': infected_history
        }
    
    def apply_hmic_strategy(self, budget: int, iterations: int) -> Dict:
        """
        Apply HMIC hiding strategy
        
        Args:
            budget: Maximum number of edge modifications per iteration
            iterations: Number of hiding iterations
            
        Returns:
            Dictionary with results for each iteration
        """
        print(f"Applying HMIC strategy with budget={budget}, iterations={iterations}")
        
        if not self.intermediate_nodes:
            print("No intermediate nodes found. Cannot apply HMIC strategy.")
            return {}
        
        results = {}
        original_centrality = self.calculate_centrality_metrics(self.mastermind)
        
        for iteration in range(iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Sort intermediate nodes by centrality (degree centrality for simplicity)
            sorted_intermediate = sorted(
                self.intermediate_nodes,
                key=lambda x: nx.degree_centrality(self.graph)[x],
                reverse=True
            )
            
            # Process each intermediate node
            for i, intermediate_node in enumerate(sorted_intermediate):
                if i >= budget:
                    break
                
                # Disconnect mastermind from intermediate node
                if self.graph.has_edge(self.mastermind, intermediate_node):
                    self.graph.remove_edge(self.mastermind, intermediate_node)
                    print(f"  Disconnected mastermind from intermediate node {intermediate_node}")
                
                # Add new edges from intermediate node to less influential nodes in community
                community_nodes = set()
                for comm in self.communities:
                    if self.mastermind in comm:
                        community_nodes = comm
                        break
                
                # Find less influential nodes (lower degree centrality)
                less_influential = sorted(
                    community_nodes - {self.mastermind, intermediate_node},
                    key=lambda x: nx.degree_centrality(self.graph)[x]
                )
                
                # Add (budget - 1) new edges
                edges_added = 0
                for target_node in less_influential:
                    if edges_added >= budget - 1:
                        break
                    if not self.graph.has_edge(intermediate_node, target_node):
                        self.graph.add_edge(intermediate_node, target_node)
                        edges_added += 1
                        print(f"  Added edge from {intermediate_node} to {target_node}")
            
            # Calculate new centrality
            new_centrality = self.calculate_centrality_metrics(self.mastermind)
            
            # Simulate SIR epidemic
            sir_results = self.simulate_sir_epidemic([self.mastermind])
            
            results[iteration] = {
                'original_centrality': original_centrality,
                'new_centrality': new_centrality,
                'centrality_reduction': {
                    metric: original_centrality[metric] - new_centrality[metric]
                    for metric in original_centrality
                },
                'sir_results': sir_results
            }
            
            print(f"  Centrality reduction: {results[iteration]['centrality_reduction']['average']:.4f}")
            print(f"  Total infected: {sir_results['total_infected']}")
        
        return results
    
    def run_complete_experiment(self, budget: int = 20, iterations: int = 5, 
                              community_index: int = 0) -> Dict:
        """
        Run complete HMIC experiment
        
        Args:
            budget: Maximum edge modifications per iteration
            iterations: Number of hiding iterations
            community_index: Index of community to analyze
            
        Returns:
            Complete experiment results
        """
        print("=" * 60)
        print("RUNNING COMPLETE HMIC EXPERIMENT")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Detect communities
        communities = self.detect_communities_greedy_modularity()
        
        if not communities or community_index >= len(communities):
            print(f"Invalid community index {community_index}. Available communities: {len(communities)}")
            return {}
        
        target_community = communities[community_index]
        
        # Step 2: Identify mastermind
        mastermind = self.identify_mastermind(target_community)
        
        # Step 3: Identify intermediate nodes
        intermediate_nodes = self.identify_intermediate_nodes(target_community, mastermind)
        
        if not intermediate_nodes:
            print("No intermediate nodes found. Cannot proceed with HMIC strategy.")
            return {}
        
        # Step 4: Apply HMIC strategy
        hmic_results = self.apply_hmic_strategy(budget, iterations)
        
        # Step 5: Calculate final metrics
        final_centrality = self.calculate_centrality_metrics(mastermind)
        final_sir = self.simulate_sir_epidemic([mastermind])
        
        # Compile results
        experiment_results = {
            'parameters': {
                'budget': budget,
                'iterations': iterations,
                'community_index': community_index,
                'community_size': len(target_community),
                'intermediate_nodes_count': len(intermediate_nodes)
            },
            'mastermind': mastermind,
            'hmic_results': hmic_results,
            'final_metrics': {
                'centrality': final_centrality,
                'sir': final_sir
            },
            'execution_time': time.time() - start_time
        }
        
        self.results = experiment_results
        
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED")
        print("=" * 60)
        print(f"Execution time: {experiment_results['execution_time']:.2f} seconds")
        
        return experiment_results
    
    def visualize_results(self, results: Dict, save_path: str = None) -> None:
        """
        Visualize experiment results
        
        Args:
            results: Experiment results dictionary
            save_path: Path to save visualization
        """
        if not results:
            print("No results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('HMIC Algorithm Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Centrality reduction over iterations
        iterations = list(results['hmic_results'].keys())
        centrality_reductions = [
            results['hmic_results'][i]['centrality_reduction']['average']
            for i in iterations
        ]
        
        axes[0, 0].plot(iterations, centrality_reductions, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Centrality Reduction Over Iterations')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Average Centrality Reduction')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Individual centrality metrics
        metrics = ['degree', 'closeness', 'betweenness']
        original_values = [results['hmic_results'][0]['original_centrality'][m] for m in metrics]
        final_values = [results['final_metrics']['centrality'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, original_values, width, label='Before HMIC', alpha=0.8)
        axes[0, 1].bar(x + width/2, final_values, width, label='After HMIC', alpha=0.8)
        axes[0, 1].set_title('Centrality Metrics Comparison')
        axes[0, 1].set_xlabel('Centrality Metric')
        axes[0, 1].set_ylabel('Centrality Value')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(metrics)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: SIR epidemic results
        sir_results = [results['hmic_results'][i]['sir_results']['total_infected'] for i in iterations]
        axes[1, 0].plot(iterations, sir_results, 'ro-', linewidth=2, markersize=8)
        axes[1, 0].set_title('SIR Epidemic Spread Results')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Total Infected Nodes')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Network visualization (simplified)
        if self.graph and len(self.graph.nodes()) <= 100:  # Only for small networks
            pos = nx.spring_layout(self.graph, seed=self.random_seed)
            
            # Color nodes by community
            node_colors = []
            for node in self.graph.nodes():
                if node == self.mastermind:
                    node_colors.append('red')  # Mastermind
                elif node in self.intermediate_nodes:
                    node_colors.append('orange')  # Intermediate nodes
                else:
                    node_colors.append('lightblue')  # Other nodes
            
            nx.draw(self.graph, pos, node_color=node_colors, node_size=50, 
                   with_labels=False, alpha=0.7, ax=axes[1, 1])
            axes[1, 1].set_title('Network Structure (Mastermind: Red, Intermediate: Orange)')
        else:
            axes[1, 1].text(0.5, 0.5, 'Network too large to visualize\n(>100 nodes)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Network Structure')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict, file_path: str) -> None:
        """
        Save experiment results to JSON file
        
        Args:
            results: Experiment results dictionary
            file_path: Path to save results
        """
        # Convert sets to lists for JSON serialization
        def convert_sets(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_sets(results)
        
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {file_path}")
    
    def compare_with_baseline(self, results: Dict) -> Dict:
        """
        Compare HMIC results with baseline (no hiding strategy)
        
        Args:
            results: HMIC experiment results
            
        Returns:
            Comparison metrics
        """
        if not results:
            return {}
        
        # Baseline: original centrality
        baseline_centrality = results['hmic_results'][0]['original_centrality']
        baseline_sir = self.simulate_sir_epidemic([self.mastermind])
        
        # HMIC: final centrality
        hmic_centrality = results['final_metrics']['centrality']
        hmic_sir = results['final_metrics']['sir']
        
        # Calculate improvements
        centrality_improvements = {}
        for metric in baseline_centrality:
            if baseline_centrality[metric] > 0:
                improvement = (baseline_centrality[metric] - hmic_centrality[metric]) / baseline_centrality[metric] * 100
                centrality_improvements[metric] = improvement
        
        sir_improvement = 0
        if baseline_sir['total_infected'] > 0:
            sir_improvement = (baseline_sir['total_infected'] - hmic_sir['total_infected']) / baseline_sir['total_infected'] * 100
        
        comparison = {
            'centrality_improvements_percent': centrality_improvements,
            'sir_improvement_percent': sir_improvement,
            'baseline_metrics': {
                'centrality': baseline_centrality,
                'sir': baseline_sir
            },
            'hmic_metrics': {
                'centrality': hmic_centrality,
                'sir': hmic_sir
            }
        }
        
        print("\n" + "=" * 40)
        print("BASELINE COMPARISON")
        print("=" * 40)
        print(f"Average centrality reduction: {centrality_improvements.get('average', 0):.2f}%")
        print(f"SIR spread reduction: {sir_improvement:.2f}%")
        
        return comparison


def main():
    """
    Main function to demonstrate HMIC algorithm usage
    """
    print("HMIC Algorithm Demonstration")
    print("=" * 50)
    
    # Initialize algorithm
    hmic = HMICAlgorithm(random_seed=42)
    
    # Load Facebook data
    hmic.load_facebook_data("facebook_combined.txt")
    
    # Run complete experiment
    results = hmic.run_complete_experiment(budget=15, iterations=3)
    
    if results:
        # Visualize results
        hmic.visualize_results(results, save_path="hmic_results.png")
        
        # Compare with baseline
        comparison = hmic.compare_with_baseline(results)
        
        # Save results
        hmic.save_results(results, "hmic_experiment_results.json")
        
        print("\nExperiment completed successfully!")
    else:
        print("Experiment failed. Check the data and parameters.")


if __name__ == "__main__":
    main()

