import numpy as np
from draw import *

# total number of observations
OBS_SIZE = 20
# length of the field i.e. LxL field
N_BATCH = 50

L = 8

# ------------------------------------------------------------------ helpers

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
  return [np.random.randint(0,2) for _ in range(L)]

def mk_query(X):
  def query(O):
    Ox, Oy = O
    if X[Ox] == X[Oy]:
        return [1.0, 0.0]
    return [0.0, 1.0]
  return query

def gen_O(X):
  query = mk_query(X)
  Ox = np.random.randint(0, L)
  Oy = np.random.randint(0, L)
  O = (Ox, Oy)
  return O, query(O) 
  
# data of the form of
#    A: the answer we're trying to infer
#    obs: the OBS_SIZE number of observations
#           divided into obs_x and obs_y
#    obs_tfs: the true/false of these observations
# all variables are a list of tensors of dimention [n_batch x ...]   
def gen_data(n_batch = N_BATCH):
  # Answer
  new_ob_x = []
  new_ob_y = []
  new_ob_tf = []
  
  obs_x = [[] for i in range(OBS_SIZE)]
  obs_y = [[] for i in range(OBS_SIZE)]
  obs_tfs = [[] for i in range(OBS_SIZE)]

  orig_x = []

  for bb in range(n_batch):
    # generate a hidden variable X
    perm = gen_X()
    orig_x.append(perm)

    new_obb_xy, new_obb_tf = gen_O(perm)
    new_obb_x, new_obb_y = vectorize(new_obb_xy)
    new_ob_x.append(new_obb_x)
    new_ob_y.append(new_obb_y)
    new_ob_tf.append(new_obb_tf)

    # generate observations for this hidden variable x
    for ob_idx in range(OBS_SIZE):
      _ob_coord, _ob_lab = gen_O(perm)
      _ob_x, _ob_y = vectorize(_ob_coord)
      obs_x[ob_idx].append(_ob_x)
      obs_y[ob_idx].append(_ob_y)
      obs_tfs[ob_idx].append(_ob_lab)

  return  np.array(obs_x, np.float32),\
          np.array(obs_y, np.float32),\
          np.array(obs_tfs, np.float32),\
          np.array(new_ob_x, np.float32),\
          np.array(new_ob_y, np.float32),\
          np.array(new_ob_tf, np.float32),\
          orig_x 
