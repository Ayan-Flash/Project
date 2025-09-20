import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import defaultdict
from networkx.algorithms.community import greedy_modularity_communities
import json

# ======================== 
# 1. Load Graph
# ======================== 
G = nx.read_edgelist("facebook_combined.txt", create_using=nx.Graph(), nodetype=int)

# Greedy modularity communities and largest community subgraph for mastermind selection
communities = list(greedy_modularity_communities(G))
largest_comm = max(communities, key=len) if communities else set(G.nodes())
G_comm = G.subgraph(largest_comm)

# ======================== 
# 2. Identify Masterminds
# ======================== 
def find_masterminds_in_community(G_comm, top_k=1):
    """Identifies the top_k most central nodes within the largest greedy modularity community using Degree Centrality (to match HMIC)."""
    centrality = nx.degree_centrality(G_comm)
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nodes[:top_k]]

mastermind = find_masterminds_in_community(G_comm)[0]
print(f"Mastermind Identified: {mastermind}")

# ======================== 
# 3. SIR Simulation & Value
# ======================== 
def run_sir(G, beta=0.05, gamma=0.02, steps=50, initial_infected=None):
    status = {node: "S" for node in G.nodes()}
    if initial_infected is None:
        initial_infected = [random.choice(list(G.nodes()))]
    for node in initial_infected:
        status[node] = "I"

    S_curve, I_curve, R_curve = [], [], []
    for _ in range(steps):
        new_status = status.copy()
        for node in G.nodes():
            if status[node] == "I":
                for neighbor in G.neighbors(node):
                    if status[neighbor] == "S" and random.random() < beta:
                        new_status[neighbor] = "I"
                if random.random() < gamma:
                    new_status[node] = "R"
        status = new_status
        S_curve.append(sum(1 for v in status.values() if v == "S"))
        I_curve.append(sum(1 for v in status.values() if v == "I") )
        R_curve.append(sum(1 for v in status.values() if v == "R") )

    return S_curve, I_curve, R_curve

# Load SIR values if they exist
try:
    with open('sir.txt', 'r') as f:
        sir_values = json.load(f)
    sir_values = {int(k): v for k, v in sir_values.items()} # Convert keys to int
except (FileNotFoundError, json.JSONDecodeError):
    print("sir.txt not found or invalid. Running SIR for all nodes to generate it.")
    sir_values = {}
    for node in G.nodes():
        _, _, R_curve = run_sir(G, initial_infected=[node])
        sir_values[node] = R_curve[-1]
    with open('sir.txt', 'w') as f:
        json.dump(sir_values, f)
    print("sir.txt created.")


# ======================== 
# 4. Custom Hiding Algorithm (FIXED)
# ======================== 
def hide_mastermind_custom(G, mastermind, removal_fraction=0.5):
    G_hidden = G.copy()
    communities = list(greedy_modularity_communities(G_hidden))
    mastermind_community = next((c for c in communities if mastermind in c), None)

    if not mastermind_community:
        print("Mastermind not found in any community.")
        return G_hidden

    # Get neighbors of the mastermind that are in the same community
    mastermind_neighbors_in_community = {n for n in G_hidden.neighbors(mastermind) if n in mastermind_community}

    # Identify intermediate nodes among the mastermind's neighbors
    intermediate_nodes = set()
    for node in mastermind_neighbors_in_community:
        for neighbor in G_hidden.neighbors(node):
            if neighbor not in mastermind_community:
                intermediate_nodes.add(node)
                break
    
    if not intermediate_nodes:
        print("No intermediate nodes found among mastermind's neighbors. Hiding may be ineffective.")
        # As a fallback, consider all neighbors in the community as candidates for removal
        intermediate_nodes = mastermind_neighbors_in_community


    # Prioritize disconnecting from more influential intermediate nodes (higher degree)
    sorted_intermediates = sorted(list(intermediate_nodes), key=lambda n: G_hidden.degree(n), reverse=True)

    # Determine the number of edges to remove based on the fraction
    budget = int(len(sorted_intermediates) * removal_fraction)
    if budget == 0 and len(sorted_intermediates) > 0:
        budget = 1 # Ensure at least one edge is removed if possible
    print(f"Hiding budget: Removing up to {budget} edges.")

    # Find less influential nodes in the community for rewiring
    community_sir = {n: sir_values.get(n, 0) for n in mastermind_community}
    sorted_community_by_sir = sorted(community_sir.items(), key=lambda x: x[1])
    less_influential_nodes = [n for n, _ in sorted_community_by_sir if n != mastermind and n not in sorted_intermediates]

    edges_removed_count = 0
    for inter_node in sorted_intermediates:
        if edges_removed_count >= budget:
            break
        
        # --- Edge Removal: Always remove the edge between mastermind and the selected intermediate node ---
        if G_hidden.has_edge(mastermind, inter_node):
            G_hidden.remove_edge(mastermind, inter_node)
            print(f"Removed edge: ({mastermind}, {inter_node})")
            edges_removed_count += 1

            # --- Edge Addition: Rewire the intermediate node to a less influential node ---
            rewired = False
            for target_node in less_influential_nodes:
                if not G_hidden.has_edge(inter_node, target_node) and inter_node != target_node:
                    G_hidden.add_edge(inter_node, target_node)
                    print(f"Added edge: ({inter_node}, {target_node})")
                    # Remove the used node to avoid connecting multiple intermediates to the same one
                    less_influential_nodes.remove(target_node)
                    rewired = True
                    break
            if not rewired:
                print(f"Could not find a suitable node to rewire {inter_node} to.")

    return G_hidden

# ======================== 
# 5. Create Centrality Table
# ======================== 
def create_centrality_table(mastermind, centralities_before, centralities_after):
    """
    Create a formatted table showing centrality scores and their reduction.
    """
    table = []
    table.append("+----------------------------------------------------+")
    table.append("|                     HMIC Analysis                  |")
    table.append("+----------------------------------------------------+")
    table.append("|                Mastermind Node: {:<16} |".format(mastermind))
    table.append("+------------------+----------------+----------------+")
    table.append("| Centrality Metric| Before Hiding  | After Hiding   |")
    table.append("+------------------+----------------+----------------+")

    for metric in ["Degree", "Closeness", "Betweenness"]:
        before_val = centralities_before[metric]
        after_val = centralities_after[metric]
        table.append("| {:<16} | {:<14.3f} | {:<14.3f} |".format(metric, before_val, after_val))
    
    table.append("+------------------+----------------+----------------+")
    
    # Add a summary row for reduction
    table.append("| Reduction Summary (Percentage)                     |")
    table.append("+------------------+----------------+----------------+")
    
    reduction_row = "|"
    for metric in ["Degree", "Closeness", "Betweenness"]:
        before_val = centralities_before[metric]
        after_val = centralities_after[metric]
        reduction = ((before_val - after_val) / before_val) * 100 if before_val > 0 else 0
        reduction_row += " {:<14.2f}% |".format(reduction)
    table.append(reduction_row)
    table.append("+------------------+----------------+----------------+")

    return "\n".join(table)

# ======================== 
# 6. Centrality & SIR Analysis
# ======================== 
def calculate_centralities(G, node):
    # Use a larger k for betweenness centrality for more stable results on larger graphs
    k_val = min(100, len(G.nodes())-1)
    dc = nx.degree_centrality(G).get(node, 0)
    cc = nx.closeness_centrality(G).get(node, 0)
    bc = nx.betweenness_centrality(G, k=k_val, normalized=True).get(node, 0)
    return {"Degree": dc, "Closeness": cc, "Betweenness": bc}

# --- Before Hiding ---
print("\n--- Analysis Before Hiding ---")
centralities_before = calculate_centralities(G, mastermind)
for name, val in centralities_before.items():
    print(f"{name} Centrality: {val:.4f}")

_, _, R_pre = run_sir(G, initial_infected=[mastermind])
pre_sir_value = R_pre[-1]
print(f"Pre-hiding SIR value: {pre_sir_value}")

# --- Apply Hiding ---
print("\n--- Applying Hiding Algorithm ---")
G_hidden = hide_mastermind_custom(G, mastermind, removal_fraction=1.0)

# --- After Hiding ---
print("\n--- Analysis After Hiding ---")
centralities_after = calculate_centralities(G_hidden, mastermind)
for name, val in centralities_after.items():
    print(f"{name} Centrality: {val:.4f}")

_, _, R_post = run_sir(G_hidden, initial_infected=[mastermind])
post_sir_value = R_post[-1]
print(f"Post-hiding SIR value: {post_sir_value}")

# Display centrality table
print("\n" + "="*50)
print("CENTRALITY SCORES TABLE")
print("="*50)
centrality_table = create_centrality_table(mastermind, centralities_before, centralities_after)
print(centrality_table)

# ======================== 
# 7. Bar Graph (Final Impact)
# ======================== 
plt.figure(figsize=(12, 6))

# Centrality Comparison
metrics = list(centralities_before.keys())
initial_vals = list(centralities_before.values())
final_vals = list(centralities_after.values())
x = np.arange(len(metrics))
width = 0.35

ax1 = plt.subplot(1, 2, 1)
rects1 = ax1.bar(x - width/2, initial_vals, width, label='Before Hiding')
rects2 = ax1.bar(x + width/2, final_vals, width, label='After Hiding')
ax1.set_ylabel('Centrality Value')
ax1.set_title('Centrality Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()

# Add percentage change labels
def autolabel(rects, initial_vals):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        initial = initial_vals[i]
        if initial > 0:
            change = (height - initial) / initial * 100
            ax1.annotate(f'{change:.1f}%', 
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom')

autolabel(rects2, initial_vals)

# SIR Comparison
ax2 = plt.subplot(1, 2, 2)
ax2.bar(['Before', 'After'], [pre_sir_value, post_sir_value], color=['blue', 'green'])
ax2.set_ylabel('Total Recovered Nodes')
ax2.set_title('SIR Model Impact')

plt.tight_layout()
plt.savefig("comparison_charts_fixed.png")
plt.show()