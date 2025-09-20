"""
HMIC (Hide Mastermind using an Intermediate Connection)
Enhanced implementation focused on stronger reduction of mastermind centralities and spread.

Key upgrades:
- Automatically detects the mastermind from the largest greedy-modularity community (no manual ID).
- Remove MULTIPLE high-impact connections per iteration (configurable "budget").
- Rewire each removed intermediate to several low-influence nodes (configurable).
- Add bypass edges among mastermind's neighbors to create alternative shortest paths (reduces betweenness).
- Output only actual, measured results (no hardcoded ROAM/HMIC paper numbers).
- Save a combined figure with centrality bars and SIR curves.
"""

import os
import random
from typing import List, Set, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class HMICAlgorithm:
    def __init__(self, graph: nx.Graph, budget: int = 5):
        """
        Args:
            graph: NetworkX graph representing social network
            budget: Number of mastermind-incident edges to remove per iteration
        Notes:
            For each removed edge (mastermind, u), we rewire u to low-influence nodes.
        """
        self.graph = graph.copy()
        self.original_graph = graph.copy()
        self.budget = budget
        self.communities: List[Set[int]] = []
        self.mastermind: Optional[int] = None
        self.hidden_community: Set[int] = set()
        self.intermediate_nodes: Set[int] = set()
        self.iteration_results: List[Dict] = []

    def detect_communities_greedy_modularity(self) -> List[Set[int]]:
        communities = list(nx.community.greedy_modularity_communities(self.graph))
        self.communities = [set(c) for c in communities]
        return self.communities

    def calculate_centralities(self, nodes: Set[int], use_full_graph: bool = True) -> Dict[int, Dict[str, float]]:
        centralities: Dict[int, Dict[str, float]] = {}
        graph_to_use = self.graph if use_full_graph else self.graph.subgraph(self.hidden_community)

        # Approximate betweenness on large graphs for speed
        k_approx = None
        if graph_to_use.number_of_nodes() > 1000:
            k_approx = min(200, max(50, int(0.05 * graph_to_use.number_of_nodes())))

        degree_cent = nx.degree_centrality(graph_to_use)
        closeness_cent = nx.closeness_centrality(graph_to_use)
        if k_approx and use_full_graph:
            betweenness_cent = nx.betweenness_centrality(graph_to_use, k=k_approx, seed=42)
        else:
            betweenness_cent = nx.betweenness_centrality(graph_to_use)

        for node in nodes:
            centralities[node] = {
                "degree": degree_cent.get(node, 0.0),
                "closeness": closeness_cent.get(node, 0.0),
                "betweenness": betweenness_cent.get(node, 0.0),
            }
        return centralities

    def identify_mastermind(self, community: Set[int], centrality_type: str = "degree") -> int:
        self.hidden_community = community
        cent = self.calculate_centralities(community, use_full_graph=False)
        mastermind = max(cent.keys(), key=lambda x: cent[x][centrality_type])
        self.mastermind = mastermind
        return mastermind

    def find_intermediate_nodes(self) -> Set[int]:
        """
        Intermediate nodes: inside hidden community, connected to mastermind,
        and having at least one neighbor outside the community (bridging nodes).
        """
        intermediate: Set[int] = set()
        external = set(self.graph.nodes()) - self.hidden_community
        for node in self.hidden_community:
            if node == self.mastermind:
                continue
            if not self.graph.has_edge(self.mastermind, node):
                continue
            nbrs = set(self.graph.neighbors(node))
            if nbrs & external:
                intermediate.add(node)
        self.intermediate_nodes = intermediate
        return intermediate

    def _intermediate_impact_scores(self, candidates: Set[int]) -> Dict[int, float]:
        """
        Heuristic impact score for an intermediate node u:
            score(u) = degree(u) * (1 + external_degree(u))
        Larger score => more valuable to disconnect.
        """
        external = set(self.graph.nodes()) - self.hidden_community
        scores: Dict[int, float] = {}
        for u in candidates:
            deg_u = float(self.graph.degree(u))
            ext_deg = sum(1 for v in self.graph.neighbors(u) if v in external)
            scores[u] = deg_u * (1.0 + float(ext_deg))
        return scores

    def get_less_influential_nodes(self, exclude_nodes: Set[int], count: int) -> List[int]:
        """
        Least-influential nodes (by degree centrality inside community) to rewire toward.
        """
        available = self.hidden_community - exclude_nodes - {self.mastermind}
        if count <= 0 or not available:
            return []
        cent = self.calculate_centralities(available, use_full_graph=False)
        sorted_nodes = sorted(available, key=lambda x: cent[x]["degree"])  # ascending degree
        return sorted_nodes[: min(count, len(sorted_nodes))]

    def _add_bypass_edges(self, count: int) -> List[Tuple[int, int]]:
        """
        Connect pairs of mastermind's neighbors (within the community) to create
        alternate routes that bypass the mastermind, reducing its betweenness.
        """
        added: List[Tuple[int, int]] = []
        if count <= 0:
            return added
        neighbors_in_comm = [v for v in self.graph.neighbors(self.mastermind) if v in self.hidden_community]
        if len(neighbors_in_comm) < 2:
            return added
        attempts = 0
        max_attempts = count * 10
        while len(added) < count and attempts < max_attempts:
            a, b = random.sample(neighbors_in_comm, 2)
            if a != b and not self.graph.has_edge(a, b):
                self.graph.add_edge(a, b)
                added.append((a, b))
            attempts += 1
        return added

    def apply_hiding_strategy(
        self,
        centrality_type: str = "degree",
        iterations: int = 2,
        additions_per_removal: int = 2,
        bypass_edges_per_iter: int = 30,
    ) -> None:
        """
        Stronger HMIC variant per iteration:
        - Remove up to `self.budget` edges (mastermind, u) where u is a high-impact intermediate.
        - For each removed u, connect u to several least-influential nodes in the community.
        - Add bypass edges among mastermind neighbors to reduce shortest-path reliance on mastermind.
        """
        self.iteration_results = []
        for it in range(iterations):
            # 1) choose top-k intermediates to cut
            candidates = self.find_intermediate_nodes()
            if not candidates:
                break
            scores = self._intermediate_impact_scores(candidates)
            ranked = sorted(candidates, key=lambda x: scores[x], reverse=True)
            to_cut = ranked[: min(self.budget, len(ranked))]

            removed_edges: List[Tuple[int, int]] = []
            rewired_edges: List[Tuple[int, int]] = []

            # 2) apply removals + rewires
            for u in to_cut:
                if self.graph.has_edge(self.mastermind, u):
                    self.graph.remove_edge(self.mastermind, u)
                    removed_edges.append((self.mastermind, u))
                    # Rewire u to low-influence nodes (avoid duplicates across u)
                    exclude = {u, self.mastermind}
                    targets = self.get_less_influential_nodes(exclude, additions_per_removal)
                    for t in targets:
                        if not self.graph.has_edge(u, t) and u != t:
                            self.graph.add_edge(u, t)
                            rewired_edges.append((u, t))

            # 3) add bypass edges between mastermind's neighbors
            bypass_edges = self._add_bypass_edges(bypass_edges_per_iter)

            # 4) snapshot mastermind centralities after this iteration
            cent = self.calculate_centralities({self.mastermind}, use_full_graph=True)
            self.iteration_results.append(
                {
                    "iteration": it + 1,
                    "removed_edges": removed_edges,
                    "rewired_edges": rewired_edges,
                    "bypass_edges": bypass_edges,
                    "centralities_after": cent[self.mastermind],
                }
            )

    def run_hmic(
        self,
        centrality_type: str = "degree",
        iterations: int = 2,
        target_mastermind: Optional[int] = None,
        additions_per_removal: int = 2,
        bypass_edges_per_iter: int = 30,
    ) -> Dict:
        communities = self.detect_communities_greedy_modularity()
        if target_mastermind is not None:
            self.mastermind = target_mastermind
            self.hidden_community = next((c for c in communities if self.mastermind in c), None)
            if not self.hidden_community:
                self.hidden_community = max(communities, key=len)
        else:
            # Auto-detect mastermind from the largest greedy-modularity community
            target_community = max(communities, key=len)
            self.mastermind = self.identify_mastermind(target_community, centrality_type)

        initial = self.calculate_centralities({self.mastermind}, use_full_graph=True)[self.mastermind]
        self.apply_hiding_strategy(
            centrality_type=centrality_type,
            iterations=iterations,
            additions_per_removal=additions_per_removal,
            bypass_edges_per_iter=bypass_edges_per_iter,
        )
        final = self.calculate_centralities({self.mastermind}, use_full_graph=True)[self.mastermind]

        return {
            "mastermind": self.mastermind,
            "initial_centralities": initial,
            "final_centralities": final,
            "iteration_details": self.iteration_results,
        }


class SIRModel:
    def __init__(self, graph: nx.Graph, infection_rate: float = 0.3, recovery_rate: float = 0.1):
        self.graph = graph
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate

    def simulate(self, initial_infected: List[int], time_steps: int = 80) -> Tuple[List[int], List[int], List[int]]:
        """
        Returns S(t), I(t), R(t)
        """
        states = {node: "S" for node in self.graph.nodes()}
        for node in initial_infected:
            if node in states:
                states[node] = "I"

        S_curve: List[int] = []
        I_curve: List[int] = []
        R_curve: List[int] = []

        for _ in range(time_steps):
            S_curve.append(sum(1 for s in states.values() if s == "S"))
            I_curve.append(sum(1 for s in states.values() if s == "I"))
            R_curve.append(sum(1 for s in states.values() if s == "R"))

            new_states = states.copy()
            for node, state in states.items():
                if state == "I":
                    # Recovery
                    if random.random() < self.recovery_rate:
                        new_states[node] = "R"
                    else:
                        # Infect neighbors
                        for nbr in self.graph.neighbors(node):
                            if states[nbr] == "S" and random.random() < self.infection_rate:
                                new_states[nbr] = "I"
            states = new_states

        return S_curve, I_curve, R_curve


def load_facebook_network(file_path: str) -> Optional[nx.Graph]:
    try:
        G = nx.read_edgelist(file_path, delimiter=" ", nodetype=int)
        if G.is_directed():
            G = G.to_undirected()
        return G
    except Exception as e:
        print(f"Error loading graph from {file_path}: {e}")
        return None


def print_hmic_table(results: Dict, nodes: int, edges: int):
    mastermind = results["mastermind"]
    initial = results["initial_centralities"]
    final = results["final_centralities"]

    header = f"HMIC Results (Facebook Network, V={nodes}, E={edges})"
    line = "=" * len(header)
    print(header)
    print(line)
    print(f"Mastermind (auto-detected): {mastermind}")
    print("-" * 60)
    print(f"{'Centrality':<12} {'Before':>12} {'After':>12} {'Change%':>10}")
    print("-" * 60)
    for k in ["degree", "closeness", "betweenness"]:
        b = initial[k]
        a = final[k]
        change = ((a - b) / b * 100) if b else 0.0
        print(f"{k.title():<12} {b:>12.4f} {a:>12.4f} {change:>9.2f}%")


def plot_hmic_graphs(
    results: Dict,
    original_graph: nx.Graph,
    modified_graph: nx.Graph,
    output_path: str = "sir_comparison_hmic_final.png",
):
    mastermind = results["mastermind"]
    initial = results["initial_centralities"]
    final = results["final_centralities"]

    # SIR curves pre/post
    sir_pre = SIRModel(original_graph, infection_rate=0.05, recovery_rate=0.02)
    sir_post = SIRModel(modified_graph, infection_rate=0.05, recovery_rate=0.02)
    _, I_pre, R_pre = sir_pre.simulate([mastermind], time_steps=100)
    _, I_post, R_post = sir_post.simulate([mastermind], time_steps=100)

    plt.figure(figsize=(12, 6))

    # Centrality bar chart
    ax1 = plt.subplot(1, 2, 1)
    metrics = ["Degree", "Closeness", "Betweenness"]
    before_vals = [initial["degree"], initial["closeness"], initial["betweenness"]]
    after_vals = [final["degree"], final["closeness"], final["betweenness"]]
    x = np.arange(len(metrics))
    width = 0.35
    ax1.bar(x - width / 2, before_vals, width, label="Before")
    ax1.bar(x + width / 2, after_vals, width, label="After")
    ax1.set_title("Mastermind Centrality (Before vs After)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylabel("Value")
    ax1.legend()

    # SIR curves
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(R_pre, label="Recovered (Pre)")
    ax2.plot(R_post, label="Recovered (Post)")
    ax2.plot(I_pre, label="Infected (Pre)", linestyle=":")
    ax2.plot(I_post, label="Infected (Post)", linestyle=":")
    ax2.set_title(f"SIR Curves (seed={mastermind})")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Count")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    # 1) Load network
    graph_path = "facebook_combined.txt"
    G = load_facebook_network(graph_path)
    if not G:
        raise SystemExit(f"Could not load graph: {graph_path}")

    # 2) Configure for stronger reduction (auto mastermind detection)
    params = {
        "budget": 12,                 # remove up to 12 mastermind edges per iteration
        "iterations": 3,              # more iterations
        # no target_mastermind: auto-detected via greedy modularity + centrality
        "additions_per_removal": 2,   # rewire per removal
        "bypass_edges_per_iter": 60,  # add many bypass edges among neighbors
    }

    hmic = HMICAlgorithm(G, budget=params["budget"])
    results = hmic.run_hmic(
        centrality_type="degree",
        iterations=params["iterations"],
        target_mastermind=None,  # ensure auto-detection path
        additions_per_removal=params["additions_per_removal"],
        bypass_edges_per_iter=params["bypass_edges_per_iter"],
    )

    # 3) Report actual measured values only
    print()
    print_hmic_table(results, nodes=G.number_of_nodes(), edges=G.number_of_edges())

    # 4) Plot actual graphs (centrality bars + SIR curves)
    out_file = "sir_comparison_hmic_final.png"
    plot_hmic_graphs(results, hmic.original_graph, hmic.graph, output_path=out_file)
    print(f"\nSaved plot: {out_file}")
