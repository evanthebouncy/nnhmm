import numpy as np
from draw import *
from world import *

# total number of observations
L = 10
OBS_SIZE = 4
# length of the field i.e. LxL field
N_BATCH = 20

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
def vectorize(coords, lengthh):
  retX = np.zeros([lengthh])
  retX[coords] = 1.0
  return retX
  

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
  return np.random.randint(0, L)

def mk_query(X):
  def query(O):
    if O < X:
      return [1.0, 0.0]
    else:
      return [0.0, 1.0]
  return query

def gen_O(X, ll = L):
  query = mk_query(X)
  O = np.random.randint(0, L)
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
  x = []
  obs_x = [[] for i in range(OBS_SIZE)]
  obs_tfs = [[] for i in range(OBS_SIZE)]
  new_ob_x = []
  new_ob_tf = []

  for bb in range(n_batch):
    # generate a hidden variable X
    _x = gen_X()
    _x_ = vectorize(_x, L)
    x.append(_x_) 

    # generate new observation
    _new_ob_coord, _new_ob_lab = gen_O(_x)
    _new_ob_x = vectorize(_new_ob_coord, L)
    new_ob_x.append(_new_ob_x)
    new_ob_tf.append(_new_ob_lab)

    # generate observations for this hidden variable x
    for ob_idx in range(OBS_SIZE):
      _ob_coord, _ob_lab = gen_O(_x)
      _ob_x = vectorize(_ob_coord, L)
      obs_x[ob_idx].append(_ob_x)
      obs_tfs[ob_idx].append(_ob_lab)


  return  np.array(x, np.float32),\
          np.array(obs_x, np.float32),\
          np.array(obs_tfs, np.float32),\
          np.array(new_ob_x, np.float32),\
          np.array(new_ob_tf, np.float32)

# get the cross entropy between truth and guess
def xentropy(truth, guess):
  return -np.sum(truth * np.log(guess))

def onehot(num, length):
  ret = np.zeros([length])
  ret[num] = 1.0
  return ret

# ------------------------------------------------------------------------------ FOR RL
class Env:
  def __init__(self, X):
    self.X = X
    self.query = mk_query(X)

  def step(self, s, a):
    a_int = np.argmax(a)
    answer = self.query(a_int)
    # have 0.0 for reward now
    return (s + [(a_int, answer)], 0.0)

  def get_final_reward(self, guess):
    true_answer = onehot(self.X, L)
    ret = -xentropy(true_answer, guess)
    return ret
    


def get_envs():
  return [Env(xx) for xx in [gen_X() for _ in range(N_BATCH)]]

def show_state(s):
  ret = []
  for stp in s:
    qq, aa = stp
    ret.append((np.argmax(qq), aa))
  return ret

