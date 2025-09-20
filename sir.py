import networkx as nx
from collections import Counter
import copy
import collections
import functools
import operator
import pickle
import json
import random

#edges = [('a','b'),('a','c'),('a','d'),('a','e'),('a','h'),('a','g'),('a','i'),('a','j'),('b','a'),('b','c'),('b','d'),('b','e'),('b','f'),('c','a'),('c','b'),('c','d'),('c','p'),('d','a'),('d','b'),('d','c'),('d','f'),('e','a'),('e','b'),('e','k'),('e','l'),('f','b'),('f','d'),('f','m'),('f','n'),('g','a'),('g','h'),('h','a'),('h','g'),('i','a'),('j','a'),('k','e'),('l','e'),('m','f'),('n','f'),('n','o'),('o','n'),('p','q'),('p','c'),('q','p'),('q','w'),('q','x'),('q','u'),('q','v'),('q','t'),('q','y'),('q','s'),('q','r')]
G = nx.read_edgelist('test.txt', create_using=nx.Graph())
nodes1 = nx.number_of_nodes(G)
print(nodes1)
print(nx.number_of_edges(G))

s = G.degree()
def Convert(tup, di):
  di = dict(tup)
  return di

deg_dictn = {}
deg_dict2n = Convert(s, deg_dictn)
#print(deg_dict2n)
def sort_dict_by_value(d, reverse=False):
  return dict(sorted(d.items(), key=lambda x: x[1], reverse=reverse))
deg_sortn = sort_dict_by_value(deg_dict2n, True)
tsum = (sum(deg_sortn.values()))
#print(tsum)

average = tsum/nodes1
print("k val is",average)

for key in deg_sortn:
    deg_sortn[key] **= 2
#print(str(deg_sortn))
tsum2 = (sum(deg_sortn.values()))
#print(tsum2)
average1 = tsum2/nodes1
print("k2 is",average1)

beta= average/(average1 - average)
print("beta value is ",beta)

INFECTED = 1
RECOVER = 2
SUSPECTED = 0
iteration = 10

def spread_infection(graph, node, infection_probability):
    infected = set()
    queue = [(node, 0)]
    while queue:
        node, hop_distance = queue.pop(0)
        if node not in infected:
            infected.add(node)
            yield node
            for neighbor in graph.neighbors(node):
                if random.random() < infection_probability:
                    queue.append((neighbor, hop_distance + 1))

result = {}
dictn = {}
dic2={}
res1={}
for j in range(iteration):
    print("Iteration number is", j)
    p = []

    for n1 in G.nodes:
        dic = {n1: 0}
        k = n1
        k = INFECTED
        for i in spread_infection(G, n1,beta):

            p.append(i)

        items = Counter(p).keys()
        p = []
        dict4 = {n1: len(items)}
        dictn.update(dict4)
        dic2.update(dic)
    my_dict = [result, dictn]
    counter = collections.Counter()
    result = dict(functools.reduce(operator.add, map(collections.Counter, my_dict)))

if len(dic2) != len(result):
    for i in dic2:
        if dic2.keys() != result.keys():
            different_keys = set(dic2.keys()) - set(result.keys())
    res1 = dict.fromkeys(different_keys, 0)
res1.update(result)
print(res1)

for key in res1:
    res1[key] /= iteration

r = {key: rank for rank, key in enumerate(sorted(set(res1.values()), reverse=True), 1)}
sorted_res1 = dict(sorted(res1.items(), key=lambda x: x[1], reverse=True))

print(sorted_res1)

with open('sir.txt','w') as file:
    file.write(json.dumps(res1))