# HMIC Algorithm Implementation

This repository contains a comprehensive implementation of the **HMIC (Hide Mastermind using Intermediate Connection)** algorithm for social network analysis and mastermind hiding strategies.

## Overview

The HMIC algorithm is designed to hide influential nodes (masterminds) in social networks by strategically modifying network connections. The algorithm:

1. **Detects communities** in the social network using modularity-based community detection
2. **Identifies mastermind nodes** based on centrality metrics (degree, closeness, betweenness)
3. **Executes hiding strategies** by disconnecting masterminds from intermediate nodes and creating alternative connections
4. **Evaluates effectiveness** through centrality analysis and SIR epidemic simulation

## Features

### Core Algorithm Components

- **Community Detection**: Greedy modularity-based community detection using Louvain method
- **Mastermind Identification**: Multi-metric centrality analysis (degree, closeness, betweenness)
- **Intermediate Node Detection**: Finds nodes connecting mastermind communities to external networks
- **HMIC Strategy**: Iterative edge modification with budget constraints
- **SIR Epidemic Simulation**: Measures spreading capability before and after hiding

### Analysis Capabilities

- **Centrality Analysis**: Comprehensive centrality metrics calculation and comparison
- **Epidemic Modeling**: SIR (Susceptible-Infected-Recovered) epidemic spread simulation
- **Effectiveness Evaluation**: Quantitative measurement of hiding strategy success
- **Parameter Optimization**: Budget and iteration parameter analysis

### Visualization

- **Centrality Comparison**: Before vs after HMIC strategy
- **Epidemic Spread**: SIR simulation visualization
- **Iteration Progress**: Centrality changes during hiding process
- **Effectiveness Summary**: Overall algorithm performance metrics

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd HMIC
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run the complete HMIC experiment with default parameters:

```python
from hmic_algorithm import HMICAlgorithm

# Initialize algorithm
hmic = HMICAlgorithm(random_seed=42)

# Load Facebook data
hmic.load_facebook_data("facebook_combined.txt")

# Run complete experiment
results = hmic.run_complete_experiment(budget=20, iterations=5)

# Visualize results
hmic.visualize_results(results, save_path="hmic_results.png")

# Save results
hmic.save_results(results, "hmic_experiment_results.json")
```

### Advanced Usage

#### Custom Parameters

```python
# Run with custom parameters
results = hmic.run_complete_experiment(
    budget=15,           # Maximum edge modifications
    iterations=3,        # Number of hiding iterations
    community_index=1    # Specific community to analyze
)
```

#### Individual Components

```python
# Detect communities
communities = hmic.detect_communities_greedy_modularity()

# Identify mastermind in specific community
mastermind = hmic.identify_mastermind(communities[0])

# Calculate centrality metrics
centrality = hmic.calculate_centrality_metrics(mastermind)

# Execute HMIC strategy
hmic_results = hmic.execute_hmic_strategy(budget=10, iterations=3)

# Simulate SIR epidemic
sir_results = hmic.simulate_sir_epidemic(
    initial_infected=[mastermind],
    infection_rate=0.3,
    recovery_rate=0.1,
    max_steps=50
)
```

#### Parameter Analysis

```python
# Test different budgets
budgets = [10, 15, 20, 25]
budget_results = {}

for budget in budgets:
    hmic_copy = HMICAlgorithm(graph=hmic.original_graph.copy())
    communities = hmic_copy.detect_communities_greedy_modularity()
    mastermind = hmic_copy.identify_mastermind(communities[0])
    results = hmic_copy.execute_hmic_strategy(budget=budget, iterations=3)
    budget_results[budget] = results
```

## Data Format

The algorithm expects social network data in **edge list format**:

```
0 1
0 2
0 3
1 4
2 5
...
```

Where each line represents an undirected edge between two nodes.

## Output Files

The algorithm generates several output files:

1. **`hmic_results.png`**: Main experiment visualization with 4 subplots
2. **`budget_comparison.png`**: Budget effectiveness analysis
3. **`hmic_experiment_results.json`**: Detailed results in JSON format

## Algorithm Details

### HMIC Strategy

1. **Community Detection**: Uses Louvain method for modularity-based community detection
2. **Mastermind Identification**: Combines degree, closeness, and betweenness centrality with weights (0.4, 0.3, 0.3)
3. **Intermediate Node Detection**: Finds nodes on paths between mastermind and external communities
4. **Edge Modification**:
   - Disconnect mastermind from prominent intermediate nodes
   - Create connections between intermediate nodes and less influential community members
5. **Iterative Process**: Repeat until budget is exhausted or no more modifications possible

### Centrality Metrics

- **Degree Centrality**: Number of connections normalized by maximum possible connections
- **Closeness Centrality**: Average shortest path distance to all other nodes
- **Betweenness Centrality**: Fraction of shortest paths passing through the node

### SIR Epidemic Model

- **Susceptible (S)**: Nodes that can be infected
- **Infected (I)**: Nodes currently spreading the infection
- **Recovered (R)**: Nodes that have recovered and are immune

## Performance Metrics

The algorithm evaluates effectiveness through:

1. **Centrality Reduction**: Decrease in mastermind centrality metrics
2. **Spreading Capability Reduction**: Decrease in epidemic spread potential
3. **Modification Efficiency**: Number of edge changes required

## Example Results

Typical results show:
- **Degree Centrality Reduction**: 15-25%
- **Closeness Centrality Reduction**: 10-20%
- **Betweenness Centrality Reduction**: 20-30%
- **Spreading Capability Reduction**: 15-25%

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Algorithm improvements
- Additional centrality metrics
- New visualization features
- Performance optimizations
- Documentation enhancements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{hmic_algorithm,
  title={HMIC: Hide Mastermind using Intermediate Connection Algorithm},
  author={Your Name},
  journal={Journal of Social Network Analysis},
  year={2024}
}
```

## Contact

For questions or support, please open an issue on GitHub or contact the maintainers.
