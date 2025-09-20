# Graph Input Formats for HMIC Algorithm

Place your graph files in the `/f:/PROJECT_020/` directory (same folder as `hmic_algorithm.py`).

## Supported Formats

### 1. Edge List (.txt or .csv)
**Format:** Each line contains two nodes representing an edge
```
1,2
1,3
2,4
3,4
4,5
```

**Usage:**
```python
G = load_facebook_network("your_edges.txt")
```

### 2. Weighted Edge List
**Format:** Each line contains node1, node2, weight
```
1,2,0.5
1,3,0.8
2,4,0.3
```

### 3. Space-separated Edge List
```
1 2
1 3
2 4
3 4
```

### 4. GML Format (.gml)
```
graph [
  node [
    id 1
    label "Node1"
  ]
  node [
    id 2
    label "Node2"
  ]
  edge [
    source 1
    target 2
  ]
]
```

### 5. GraphML (.graphml)
XML-based format that preserves node and edge attributes.

### 6. Adjacency List (.adjlist)
```
1 2 3
2 1 4
3 1 4
4 2 3 5
```

## Facebook Dataset Format
If you have Facebook network data, it's typically in edge list format:
```
0 1
0 2
0 3
1 4
...
```

## Quick Test
To test with your graph:

1. Put your graph file in `/f:/PROJECT_020/`
2. Update the code:
```python
# Replace this line in hmic_algorithm.py
G = load_facebook_network("your_graph_file.txt")
```

## Example Files
You can create test files like:

**small_network.txt:**
```
1,2
2,3
3,4
4,5
5,1
1,3
2,4
3,5
```

Then use:
```python
G = load_facebook_network("small_network.txt")
```
