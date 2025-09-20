import networkx as nx
from hmic_algorithm import HMICAlgorithm, load_facebook_network
from networkx.algorithms.community import greedy_modularity_communities

G = load_facebook_network("facebook_combined.txt")
h = HMICAlgorithm(G, budget=20)
communities = h.detect_communities_greedy_modularity()
largest = max(communities, key=len)
mm_h = h.identify_mastermind(largest, centrality_type='degree')

# graph.py logic replicated
communities2 = list(greedy_modularity_communities(G))
largest2 = max(communities2, key=len) if communities2 else set(G.nodes())
sub = G.subgraph(largest2)
dc = nx.degree_centrality(sub)
mm_g = max(dc.items(), key=lambda x: x[1])[0]

print(f"HMIC mastermind: {mm_h}")
print(f"Graph.py-logic mastermind: {mm_g}")
print("MATCH" if mm_h == mm_g else "MISMATCH")
