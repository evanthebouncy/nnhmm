import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

N = 20

def dist(c1, c2):
  return np.linalg.norm(np.array(c1) - np.array(c2))

def gen_pts(n):
  ret = []
  for i in range(n):
    ret.append((np.random.random(), np.random.random()))
  return ret

def gen_graph(n_vertex, c):
  n = len(n_vertex)
  ret = [[0 for i in range(n)] for j in range(n)]
  for i in range(n):
    for j in range(i):
      crd_i, crd_j = n_vertex[i], n_vertex[j]
      if np.random.random() < np.power(2, -c * dist(crd_i, crd_j)) or\
         dist(crd_i, crd_j) < 0.3:
        ret[i][j] = 1
        ret[j][i] = 1
  return ret

def get_shortest_path(ge, node_i):
  fringe = [node_i]
  seen = set()
  dists = {}
  for _ in range(N):
    dists[_] = 999

  cur_hop = 0
  while fringe != []:
    cur_nodes = fringe
    seen.update(cur_nodes)
    fringe = []
    for c_n in cur_nodes:
      dists[c_n] = cur_hop
      for other_j in range(N):
        if ge[c_n][other_j] and other_j not in seen:
          fringe.append(other_j) 
    fringe = list(set(fringe))
    # print fringe
    cur_hop += 1
  return dists
    
def get_shortest_paths(ge):
  return [get_shortest_path(ge, i) for i in range(N)]

def random_fail(ge, prob=0.4):
  ret = copy.deepcopy(ge)
  for i in range(N):
    for j in range(i):
      if np.random.random() < prob:
        ret[i][j] = 0
        ret[j][i] = 0
  return ret 

def path_changed(ge, ge_fail):
  sp_ge = get_shortest_paths(ge)
  sp_ge_fail = get_shortest_paths(ge_fail)
  ret = set()
  for i in range(N):
    for j in range(N):
      # print i, j, sp_ge[i][j] , sp_ge_fail[i][j]
      if sp_ge[i][j] != sp_ge_fail[i][j]:
        ret.add((i,j))
  return ret  

def draw_graph(gv, ge, name):
  Gr = nx.Graph()
  for i in range(N):
    Gr.add_node(i, pos=gv[i])

  for i in range(N):
    for j in range(N):
      if ge[i][j]:
        Gr.add_edge(i,j)

  labels = dict()
  for i in range(N):
    labels[i] = str(i)

  pos=nx.get_node_attributes(Gr,'pos')

  nx.draw(Gr, pos=pos, 
      node_size=400, with_labels=False)
  nx.draw_networkx_labels(Gr, pos, labels)

  plt.savefig(name)





# V = gen_pts(N)
# G = gen_graph(V, 6)
# 
# G_V = V
# G_E = G
# 
# print G_V
# print G_E
# 
# Gr = nx.Graph()
# for i in range(N):
#   Gr.add_node(i, pos=G_V[i])
# 
# for i in range(N):
#   for j in range(N):
#     if G_E[i][j]:
#       Gr.add_edge(i,j)
# 
# labels = dict()
# for i in range(N):
#   labels[i] = str(i)
# 
# pos=nx.get_node_attributes(Gr,'pos')
# 
# nx.draw(Gr, pos=pos, 
#     node_size=400, with_labels=False)
# nx.draw_networkx_labels(Gr, pos, labels)
# 
# plt.savefig("graph.png")
