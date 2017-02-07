import numpy as np
import matplotlib.pylab as plt
import multiprocessing as mp

from matplotlib import figure

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
  ret_shape = [10, 10, 1]
  ret = np.zeros(shape=ret_shape)
  for ob, lab in obs:
    ii, jj = ob
    labb = 1.0 if lab[0] > lab[1] else -1.0
    ret[ii][jj][0] = labb
  draw(ret, name)

def draw_annotate(x_cords, y_cords, anns, name):
  FIG.clf()
  y = x_cords
  z = y_cords
  n = anns
  fig = FIG
  ax = fig.add_subplot(1,1,1)
  ax.set_xlim([0,10])
  ax.set_ylim([0,10])
  ax.set_ylim(ax.get_ylim()[::-1])
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
      print ob
      ob_coord, ob_outcome = ob
      x_coords.append(ob_coord[0])
      y_coords.append(ob_coord[1])
      anno.append("O"+str(i)+str(int(ob_outcome[0])))

  draw_annotate(x_coords, y_coords, anno, name)

def draw_ship(ship_coords, name):   
  ret_shape = [10, 10, 1]
  ret = np.zeros(shape=ret_shape)

  for sc in ship_coords:
    l = sc[0][0]
    u = sc[0][1]
    r = sc[1][0]
    d = sc[1][1]
    for i in range(l, r+1):
      for j in range(u, d+1):
        ret[i][j][0] = 1.0
  
  draw(ret, name)

def draw_all_preds(all_preds, name):
  # draw again for predictions
  ret_shape = [8, 8, 1]
  ret = np.zeros(shape=ret_shape)

  for qq, labb in all_preds:
    i, j = qq
    ret[i][j][0] = labb[0]
  
  draw(ret, name)

def draw_truth(qry, name):
  ret_shape = [8, 8, 1]
  ret = np.zeros(shape=ret_shape)
  all_preds = [(i,j) for i in range(8) for j in range(8)]

  for qq in all_preds:
    labb = qry(qq)
    i, j = qq
    ret[i][j][0] = labb[0]
  
  draw(ret, name)











 
