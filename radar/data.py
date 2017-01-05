import numpy as np
from draw import *

# total number of observations
OBS_SIZE = 10
# length of the field i.e. LxL field
L = 20
N_BATCH = 50

# ------------------------------------------------------------------ helpers

# pad a list into its final length and also turn it into a numpy array
def pad_and_numpy(lst, final_length):
  # convert
  lst = [np.array(x, np.float32) for x in lst]
  length = len(lst)
  # crop if too long
  if length > final_length:
    return lst[:final_length]
  item_shape = lst[0].shape
  return lst + [np.zeros(item_shape, np.float32) for i in range(final_length - length)]

def dist(pt1, pt2):
  x1, y1 = pt1
  x2, y2 = pt2
  return np.sqrt(np.power(x1-x2,2) + np.power(y1-y2,2))

def coord_2_loc(coord, ll = L):
  ret = np.zeros([ll, ll, 1])
  for i in range(ll):
    for j in range(ll):
      grid_coord_i = i + 0.5
      grid_coord_j = j + 0.5
      ret[i][j][0] = np.exp(-dist((grid_coord_i, grid_coord_j), coord) / 1.0)

  # ssums = ret.sum()
  # ret = ret / ssums
  return ret

def coord_2_loc_obs(coord, label, ll = L):
  idxx = 0 if label[0] == 1.0 else 1
  ret = np.zeros([ll, ll, 2])
  for i in range(ll):
    for j in range(ll):
      grid_coord_i = i + 0.5
      grid_coord_j = j + 0.5
      ret[i][j][idxx] = np.exp(-dist((grid_coord_i, grid_coord_j), coord) / 1.0)
  return ret

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
def gen_X(ll = L):
  x_coord = np.random.randint(0, ll)
  y_coord = np.random.randint(0, ll)
  return x_coord, y_coord

def mk_query(X, ll=L):
  def query(O):
    if dist(X,O) < ll / 3.0:
      return [1.0, 0.0]
    else:
      return [0.0, 1.0]
  return query

def gen_O(X, ll = L):
  query = mk_query(X)
  Ox = np.random.randint(0, ll)
  Oy = np.random.randint(0, ll)
  O = (Ox, Oy)
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
  x_y = []
  obs_x = [[] for i in range(OBS_SIZE)]
  obs_y = [[] for i in range(OBS_SIZE)]
  obs_tfs = [[] for i in range(OBS_SIZE)]
  new_ob_x = []
  new_ob_y = []
  new_ob_tf = []

  for bb in range(n_batch):
    # generate a hidden variable X
    x_coord = gen_X()
    _x_x, _x_y = vectorize(x_coord)
    x_x.append(_x_x) 
    x_y.append(_x_y) 

    # generate new observation
    _new_ob_coord, _new_ob_lab = gen_O(x_coord)
    _new_ob_x, _new_ob_y = vectorize(_new_ob_coord)
    new_ob_x.append(_new_ob_x)
    new_ob_y.append(_new_ob_y)
    new_ob_tf.append(_new_ob_lab)

    # generate observations for this hidden variable x
    for ob_idx in range(OBS_SIZE):
      _ob_coord, _ob_lab = gen_O(x_coord)
      _ob_x, _ob_y = vectorize(_ob_coord)
      obs_x[ob_idx].append(_ob_x)
      obs_y[ob_idx].append(_ob_y)
      obs_tfs[ob_idx].append(_ob_lab)


  return  np.array(x_x, np.float32),\
          np.array(x_y, np.float32),\
          np.array(obs_x, np.float32),\
          np.array(obs_y, np.float32),\
          np.array(obs_tfs, np.float32),\
          np.array(new_ob_x, np.float32),\
          np.array(new_ob_y, np.float32),\
          np.array(new_ob_tf, np.float32)

