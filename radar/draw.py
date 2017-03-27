import numpy as np
import matplotlib.pylab as plt
import multiprocessing as mp

from matplotlib import figure

# m = [[0.0, 1.47, 2.43, 3.44, 1.08, 2.83, 1.08, 2.13, 2.11, 3.7], [1.47, 0.0, 1.5,     2.39, 2.11, 2.4, 2.11, 1.1, 1.1, 3.21], [2.43, 1.5, 0.0, 1.22, 2.69, 1.33, 3.39, 2.15, 2.12, 1.87], [3.44, 2.39, 1.22, 0.0, 3.45, 2.22, 4.34, 2.54, 3.04, 2.28], [1.08, 2.11, 2.69, 3.45, 0.0, 3.13, 1.76, 2.46, 3.02, 3.85], [2.83, 2.4, 1.33, 2.22, 3.13, 0.0, 3.83, 3.32, 2.73, 0.95], [1.08, 2.11, 3.39, 4.34, 1.76, 3.83, 0.0, 2.47, 2.44, 4.74], [2.13, 1.1, 2.15, 2.54, 2.46, 3.32, 2.47, 0.0, 1.78, 4.01], [2.11, 1.1, 2.12, 3.04, 3.02, 2.73, 2.44, 1.78, 0.0, 3.57], [3.7, 3.21, 1.87, 2.28, 3.85, 0.95, 4.74, 4.01, 3.57, 0.0]]

FIG = plt.figure()

def draw_coord(coord, name, lab=[1.0, 0.0]):
  color = 1.0 if lab[0] > lab[1] else -1.0
  ret = np.zeros(shape=[20,20,1])
  coord_x, coord_y = coord
  coord_x_idx = np.argmax(coord_x)
  coord_y_idx = np.argmax(coord_y)
  ret[coord_x_idx][coord_y_idx][0] = color

  draw(ret, name)
  

def draw(m, name):
  FIG.clf()

  matrix = m
  orig_shape = np.shape(matrix)
  # lose the channel shape in the end of orig_shape
  new_shape = orig_shape[:-1] 
  matrix = np.reshape(matrix, new_shape)
  ax = FIG.add_subplot(1,1,1)
  ax.set_aspect('equal')
  plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
  # plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean)
  plt.colorbar()
  plt.savefig(name)

def draw_obs(obs, name):
  ret_shape = [20, 20, 1]
  ret = np.zeros(shape=ret_shape)
  for ii, ob in enumerate(obs):
    if ob.max() > 0.0:
      idxx = np.unravel_index(ob.argmax(), ob.shape)
      if idxx[-1] == 0:
        ret[idxx[0]][idxx[1]] = 1.0 * ii
      else:
        ret[idxx[0]][idxx[1]] = -1.0 * ii
  draw(ret, name)

def draw_annotate(x_cords, y_cords, anns, name):
  FIG.clf()
  y = x_cords
  z = y_cords
  n = anns
  fig = FIG
  ax = fig.add_subplot(1,1,1)
  ax.set_xlim([0,20])
  ax.set_ylim([0,20])
  ax.scatter(z, y)
  for i, txt in enumerate(n):
    ax.annotate(txt, (z[i],y[i]))
  fig.savefig(name)

def draw_trace(trace, name):
  x_coords = []
  y_coords = []
  anno = []
  for i, stuff in enumerate(trace):
    ob, inv = stuff
#    x_coords.append(inv[0])
#    y_coords.append(inv[1])
#    anno.append("X"+str(i))

    if ob != None:
      ob_coord, ob_outcome = ob
      x_coords.append(ob_coord[0])
      y_coords.append(ob_coord[1])
      anno.append("O"+str(i)+str(int(ob_outcome[0])))

  draw_annotate(x_coords, y_coords, anno, name)

   
















 
