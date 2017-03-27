import numpy as np
from draw import *

# total number of observations
OBS_SIZE = 10
# length of the field i.e. LxL field
L = 5
K = 2
N_BATCH = 50

# ------------------------------------------------------------------ helpers

def dist(pt1, pt2):
  x1, y1 = pt1
  x2, y2 = pt2
  return np.sqrt(np.power(x1-x2,2) + np.power(y1-y2,2))

# turn a coordinate to a pair of numpy objects
def vectorize(coords):
  retX, retY = np.zeros([L]), np.zeros([L])
  retX[coords[0]] = 1.0
  retY[coords[1]] = 1.0
  return retX, retY
  

# show dimension of a data object (list of list or a tensor)
def show_dim(lst1):
  if hasattr(lst1, '__len__') and len(lst1) > 0:
    return [len(lst1), show_dim(lst1[0])]
  else:
    try:
      return lst1.get_shape()
    except:
      try:
        return lst1.shape
      except:
        return type(lst1)

# --------------------------------------------------------------- modelings
# generate the hidden state
def gen_X():
  classes = [np.random.randint(0, K) for _ in range(L)]
  ret = np.zeros([L,L])
  for i in range(L):
    for j in range(L):
      if classes[i] == classes[j]:
        ret[i][j] = 1.0
  return ret

def mk_query(X):
  def query(O):
    ob1, ob2 = O
    if X[ob1][ob2] == 1.0:
      return [1.0, 0.0]
    else:
      return [0.0, 1.0]
  return query

def gen_O(X):
  query = mk_query(X)
  item1 = np.random.randint(0, L)
  item2 = np.random.randint(0, L)
  if item1 == item2:
    return gen_O(X)
  O = (item1, item2)
  return O, query(O) 
  
# data of the form of
#    x: the true location of the hidden variable
#           divided into xx and xy
#    obs: the OBS_SIZE number of observations
#           divided into obs_x and obs_y
#    obs_tfs: the true/false of these observations
#    new_ob: the new observation
#           divided into new_ob_x and new_ob_y
#    new_ob_tf: the true/false for new ob
# all variables are a list of tensors of dimention [n_batch x ...]   
def gen_data(n_batch = N_BATCH):
  # each element of shape [batch x loc]
  x_x = []
  obs_1 = [[] for i in range(OBS_SIZE)]
  obs_2 = [[] for i in range(OBS_SIZE)]
  obs_tfs = [[] for i in range(OBS_SIZE)]
  new_ob_1 = []
  new_ob_2 = []
  new_ob_tf = []

  for bb in range(n_batch):
    # generate a hidden variable X
    x = gen_X()
    x_x.append(x) 


    # generate observations for this hidden variable x
    for ob_idx in range(OBS_SIZE):
      _ob_item, _ob_lab = gen_O(x)
      _ob_1, _ob_2 = vectorize(_ob_item)
      obs_1[ob_idx].append(_ob_1)
      obs_2[ob_idx].append(_ob_2)
      obs_tfs[ob_idx].append(_ob_lab)

    # generate new observation
    _new_ob_items, _new_ob_lab = gen_O(x)
    _new_ob_1, _new_ob_2 = vectorize(_new_ob_items)
    new_ob_1.append(_new_ob_1)
    new_ob_2.append(_new_ob_2)
    new_ob_tf.append(_new_ob_lab)

  return  np.array(x_x, np.float32),\
          np.array(obs_1, np.float32),\
          np.array(obs_2, np.float32),\
          np.array(obs_tfs, np.float32),\
          np.array(new_ob_1, np.float32),\
          np.array(new_ob_2, np.float32),\
          np.array(new_ob_tf, np.float32)

