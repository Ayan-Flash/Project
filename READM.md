| **Aspect**                | **HMIC (Rewire)**                                                 | **Intermediate Removal**                                           |
| ------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Mastermind(s)**         | Node **51**                                                       | Nodes **22, 40**                                                   |
| **Centralities (Before)** | DC = 0.462<br>CC = 0.619<br>BC = 0.552                            | –                                                                  |
| **Centralities (After)**  | DC = 0.769<br>CC = 0.813<br>BC = 0.665                            | –                                                                  |
| **SIR Spread (Before)**   | 69 infected                                                       | Node 22 → 64<br>Node 40 → 67                                       |
| **SIR Spread (After)**    | 66 infected                                                       | Node 22 → 60<br>Node 40 → 54                                       |
| **Reduction**             | 3 infections (\~4%)                                               | Node 22: 4 infections (\~6.3%)<br>Node 40: 13 infections (\~19.4%) |
| **Hiding Operation**      | Rewired mastermind edges to weaker nodes (budget=3, 2 iterations) | Removed **nodes 58, 35** (top intermediates by SIR impact)         |



PS F:\PROJECT_020> python hmic_algorithm.py
Loading network from: facebook_combined.txt
Network loaded: 4039 nodes, 88234 edges
SUCCESS: This appears to be the Facebook dataset (4039 nodes, 88234 edges)
Network loaded: 4039 nodes, 88234 edges
=== HMIC Algorithm Execution ===

1. Detecting communities...
Found 13 communities
Selected community size: 983

2. Identifying mastermind...
Mastermind identified: 1684
Initial mastermind centralities (full graph):
  degree: 0.1961
  closeness: 0.3936
  betweenness: 0.3378
Initial mastermind centralities (community subgraph):
  degree: 0.5468
  closeness: 0.5468
  betweenness: 0.5036

3. Applying hiding strategy...

=== Iteration 1 ===
Found 46 intermediate nodes connected to mastermind
Intermediate nodes: [2946, 1666, 2952, 2826, 1419]...
Mastermind has 792 neighbors
Removed edge: 1684 - 107 (degree: 1045)
Added connection: 107 - 2842
Removed edge: 1684 - 2839 (degree: 137)
Added connection: 2839 - 3031
Removed edge: 1684 - 3363 (degree: 131)
Added connection: 3363 - 3071
Modified 3 edges in this iteration
Centralities after iteration (full graph):
  degree: 0.1954
  closeness: 0.3489
  betweenness: 0.3049
Centralities after iteration (community subgraph):
  degree: 0.5468
  closeness: 0.5468
  betweenness: 0.5036

=== Iteration 2 ===
Found 49 intermediate nodes connected to mastermind
Intermediate nodes: [2946, 1666, 2952, 2826, 1419]...
Mastermind has 789 neighbors
Removed edge: 1684 - 2754 (degree: 122)
Added connection: 2754 - 3183
Removed edge: 1684 - 3101 (degree: 122)
Added connection: 3101 - 3230
Removed edge: 1684 - 3291 (degree: 119)
Added connection: 3291 - 2788
Modified 3 edges in this iteration
Centralities after iteration (full graph):
  degree: 0.1947
  closeness: 0.3488
  betweenness: 0.3046
Centralities after iteration (community subgraph):
  degree: 0.5448
  closeness: 0.5462
  betweenness: 0.5024

Final mastermind centralities (full graph):
  degree: 0.1947
  closeness: 0.3488
  betweenness: 0.3046
Final mastermind centralities (community subgraph):
  degree: 0.5448
  closeness: 0.5462
  betweenness: 0.5024

=== CENTRALITY REDUCTION ANALYSIS ===
Mastermind hiding effectiveness:
  DEGREE centrality: 0.76% reduction (SUCCESS)
  CLOSENESS centrality: 11.39% reduction (SUCCESS)
  BETWEENNESS centrality: 9.83% reduction (SUCCESS)

Overall hiding result: SUCCESSFULLY HIDDEN (avg reduction: 7.33%) 

=== HMIC Results Summary ===
Mastermind: 1684
Hidden community size: 983
Iterations performed: 2

==================================================
CENTRALITY SCORES TABLE
==================================================
┌─────────────────────────────────────┐
│                HMIC                 │
├─────────────────────────────────────┤
│                1684                │
├─────────────┬─────────────┬─────────┤
│     DC      │     CC      │    BC   │
├─────────────┼─────────────┼─────────┤
│   0.196    │   0.394    │  0.338  │
│   0.195    │   0.349    │  0.305  │
│   0.195    │   0.349    │  0.305  │
└─────────────┴─────────────┴─────────┘

Centrality changes:
  degree: 0.1961 -> 0.1947 (0.76% reduction) ✓ HIDDEN
  closeness: 0.3936 -> 0.3488 (11.39% reduction) ✓ HIDDEN
  betweenness: 0.3378 -> 0.3046 (9.83% reduction) ✓ HIDDEN        

=== SIR Model Evaluation ===
Max infected nodes - Original: 2651
Max infected nodes - Modified: 2580
Infection reduction: 2.68%

=== HMIC Algorithm Completed ===


---------------------------------------------------------------------------------------------------------------------------
PS F:\PROJECT_020> python hmic_algorithm.py
Loading network from: facebook_combined.txt
Network loaded: 4039 nodes, 88234 edges
SUCCESS: This appears to be the Facebook dataset (4039 nodes, 88234 edges)
Network loaded: 4039 nodes, 88234 edges
=== HMIC Algorithm Execution ===

1. Detecting communities...
Found 13 communities
Selected community size: 983

2. Identifying mastermind...
Mastermind identified: 1684
Initial mastermind centralities (full graph):
  degree: 0.1961
  closeness: 0.3936
  betweenness: 0.3378
Initial mastermind centralities (community subgraph):
  degree: 0.5468
  closeness: 0.5468
  betweenness: 0.5036

3. Applying hiding strategy...

=== Iteration 1 ===
Found 46 intermediate nodes connected to mastermind
Intermediate nodes: [2946, 1666, 2952, 2826, 1419]...
Mastermind has 792 neighbors
Removed edge: 1684 - 107 (degree: 1045)
Added connection: 107 - 2842
Removed edge: 1684 - 2839 (degree: 137)
Added connection: 2839 - 3031
Removed edge: 1684 - 3363 (degree: 131)
Added connection: 3363 - 3071
Removed edge: 1684 - 2754 (degree: 122)
Added connection: 2754 - 3183
Removed edge: 1684 - 3101 (degree: 122)
Added connection: 3101 - 3230
Removed edge: 1684 - 3291 (degree: 119)
Added connection: 3291 - 2788
Removed edge: 1684 - 2742 (degree: 116)
Added connection: 2742 - 2842
Removed edge: 1684 - 3082 (degree: 116)
Added connection: 3082 - 2857
Removed edge: 1684 - 3397 (degree: 115)
Added connection: 3397 - 3006
Removed edge: 1684 - 3320 (degree: 113)
Added connection: 3320 - 3031
Removed edge: 1684 - 3426 (degree: 113)
Added connection: 3426 - 3071
Removed edge: 1684 - 3090 (degree: 108)
Added connection: 3090 - 3134
Removed edge: 1684 - 2966 (degree: 107)
Added connection: 2966 - 3183
Removed edge: 1684 - 3280 (degree: 107)
Added connection: 3280 - 3230
Removed edge: 1684 - 3434 (degree: 107)
Added connection: 3434 - 3268
Removed edge: 1684 - 2944 (degree: 106)
Added connection: 2944 - 3282
Removed edge: 1684 - 3232 (degree: 105)
Added connection: 3232 - 3408
Removed edge: 1684 - 2863 (degree: 104)
Added connection: 2863 - 3407
Removed edge: 1684 - 2877 (degree: 104)
Added connection: 2877 - 3125
Removed edge: 1684 - 3116 (degree: 103)
Added connection: 3116 - 1854
Modified 20 edges in this iteration
Centralities after iteration (full graph):
  degree: 0.1912
  closeness: 0.3483
  betweenness: 0.3022
Centralities after iteration (community subgraph):
  degree: 0.5367
  closeness: 0.5443
  betweenness: 0.4960

Progress towards target values:
  degree: 0.1912 (target: 0.156) - Need 0.0352 more reduction
  closeness: 0.3483 (target: 0.098) - Need 0.2503 more reduction
  betweenness: 0.3022 (target: 0.296) - Need 0.0062 more reduction

=== Iteration 2 ===
Found 53 intermediate nodes connected to mastermind
Intermediate nodes: [2946, 1666, 2952, 2826, 1419]...
Mastermind has 772 neighbors
Removed edge: 1684 - 2786 (degree: 102)
Added connection: 2786 - 2691
Removed edge: 1684 - 2951 (degree: 102)
Added connection: 2951 - 3375
Removed edge: 1684 - 3051 (degree: 101)
Added connection: 3051 - 2714
Removed edge: 1684 - 2719 (degree: 100)
Added connection: 2719 - 2722
Removed edge: 1684 - 2986 (degree: 100)
Added connection: 2986 - 2788
Removed edge: 1684 - 2661 (degree: 99)
Added connection: 2661 - 2792
Removed edge: 1684 - 2793 (degree: 99)
Added connection: 2793 - 2808
Removed edge: 1684 - 3078 (degree: 99)
Added connection: 3078 - 2842
Removed edge: 1684 - 2782 (degree: 98)
Added connection: 2782 - 2857
Removed edge: 1684 - 2669 (degree: 97)
Added connection: 2669 - 2860
Removed edge: 1684 - 3387 (degree: 97)
Added connection: 3387 - 2922
Removed edge: 1684 - 2956 (degree: 96)
Added connection: 2956 - 899
Removed edge: 1684 - 3026 (degree: 96)
Added connection: 3026 - 2952
Removed edge: 1684 - 2716 (degree: 95)
Added connection: 2716 - 2975
Removed edge: 1684 - 2778 (degree: 95)
Added connection: 2778 - 2998
Modified 15 edges in this iteration
Centralities after iteration (full graph):
  degree: 0.1875
  closeness: 0.3479
  betweenness: 0.2967
Centralities after iteration (community subgraph):
  degree: 0.5295
  closeness: 0.5422
  betweenness: 0.4916

Progress towards target values:
  degree: 0.1875 (target: 0.156) - Need 0.0315 more reduction
  closeness: 0.3479 (target: 0.098) - Need 0.2499 more reduction
  betweenness: 0.2967 (target: 0.296) - Need 0.0007 more reduction

Final mastermind centralities (full graph):
  degree: 0.1875
  closeness: 0.3479
  betweenness: 0.2967
Final mastermind centralities (community subgraph):
  degree: 0.5295
  closeness: 0.5422
  betweenness: 0.4916

=== CENTRALITY REDUCTION ANALYSIS ===
Mastermind hiding effectiveness:
  DEGREE centrality: 4.42% reduction (SUCCESS)
  CLOSENESS centrality: 11.61% reduction (SUCCESS)
  BETWEENNESS centrality: 12.17% reduction (SUCCESS)

Overall hiding result: SUCCESSFULLY HIDDEN (avg reduction: 9.40%)

=== HMIC Results Summary ===
Mastermind: 1684
Hidden community size: 983
Iterations performed: 2

==================================================
CENTRALITY SCORES TABLE
==================================================
┌─────────────────────────────────────┐
│                HMIC                 │
├─────────────────────────────────────┤
│                1684                │
├─────────────┬─────────────┬─────────┤
│     DC      │     CC      │    BC   │
├─────────────┼─────────────┼─────────┤
│   0.196    │   0.394    │  0.338  │
│   0.187    │   0.348    │  0.297  │
│   0.187    │   0.348    │  0.297  │
└─────────────┴─────────────┴─────────┘

Centrality changes:
  degree: 0.1961 -> 0.1875 (4.42% reduction) ✓ HIDDEN
  closeness: 0.3936 -> 0.3479 (11.61% reduction) ✓ HIDDEN
  betweenness: 0.3378 -> 0.2967 (12.17% reduction) ✓ HIDDEN

=== SIR Model Evaluation ===
Max infected nodes - Original: 2739
Max infected nodes - Modified: 2853
Infection reduction: -4.16%

=== HMIC Algorithm Completed ===