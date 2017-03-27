import numpy as np
from draw import *

# total number of observations
OBS_SIZE = 10
# length of the field i.e. LxL field
N_BATCH = 50

RAND_HIT = 0.4

L = 5
S1DIM = (1,4)
S2DIM = (2,2)

# L = 10
# S1DIM = (2,5)
# S2DIM = (1,8)

# ------------------------------------------------------------------ helpers

# turn a coordinate to a pair of numpy objects
def vectorize(coords):
  retX, retY = np.zeros([L]), np.zeros([L])
  retX[coords[0]] = 1.0
  retY[coords[1]] = 1.0
  return retX, retY
  
def in_bound(pt):
  x, y = pt
  return 0 <= x < L and 0 <= y < L

# 2 rectangles do NOT overlap if there's a horizontal line
# or a vertical line that can divide the two apart
def rec_not_overlap(rec1, rec2):
  l1, l2 = rec1[0][0], rec2[0][0]
  u1, u2 = rec1[0][1], rec2[0][1]
  r1, r2 = rec1[1][0], rec2[1][0]
  d1, d2 = rec1[1][1], rec2[1][1]

  exist_vert_line = r1 < l2 or r2 < l1
  exist_hori_line = d1 < u2 or d2 < u1

  ret = exist_vert_line or exist_hori_line
  return ret

def is_legal(ship_coords):
  # check dup
  if len(set(ship_coords)) != len(ship_coords): return False
  for sc in ship_coords:
    ul, lr, _ = sc
    if not in_bound(ul):
      return False
    if not in_bound(lr):
      return False

  for sc1 in ship_coords:
    for sc2 in ship_coords:
      if sc1 != sc2:
        if not rec_not_overlap(sc1, sc2):
          return False
  return True
      
  
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
  sc1_orient = np.random.random() > 0.5
  sub1_x = S1DIM[0] if sc1_orient else S1DIM[1]
  sub1_y = S1DIM[1] if sc1_orient else S1DIM[0]
  sc1_ul = np.random.randint(0, L-sub1_x+1), np.random.randint(0, L-sub1_y+1)
  sc1_lr = (sc1_ul[0] + S1DIM[0]-1, sc1_ul[1] + S1DIM[1]-1) if sc1_orient\
            else (sc1_ul[0] + S1DIM[1]-1, sc1_ul[1] + S1DIM[0]-1)

  sc2_orient = np.random.random() > 0.5
  sub2_x = S2DIM[0] if sc2_orient else S2DIM[1]
  sub2_y = S2DIM[1] if sc2_orient else S2DIM[0]
  sc2_ul = np.random.randint(0, L-sub2_x+1), np.random.randint(0, L-sub2_y+1)
  sc2_lr = (sc2_ul[0] + S2DIM[0]-1, sc2_ul[1] + S2DIM[1]-1) if sc2_orient\
            else (sc2_ul[0] + S2DIM[1]-1, sc2_ul[1] + S2DIM[0]-1)

  sc1_orient_b = (1.0, 0.0) if sc1_orient else (0.0, 1.0)
  sc2_orient_b = (1.0, 0.0) if sc2_orient else (0.0, 1.0)
  ship_coords = [(sc1_ul, sc1_lr, sc1_orient_b), (sc2_ul, sc2_lr, sc2_orient_b)]
  if is_legal(ship_coords):
    return ship_coords
  else:
    return gen_X()


def mk_query(X):
  def query(O):
    Ox, Oy = O
    for sc in X:
      l = sc[0][0]
      u = sc[0][1]
      r = sc[1][0]
      d = sc[1][1]
      if l <= Ox <= r and u <= Oy <= d:
        return [1.0, 0.0]
    return [0.0, 1.0]
  return query

def get_all_ship_coords(X):
  ret = []
  for sc in X:
    l = sc[0][0]
    u = sc[0][1]
    r = sc[1][0]
    d = sc[1][1]
    for i in range(l, r+1):
      for j in range(u, d+1):
        ret.append((i,j))
  return ret

def gen_O(X):
  query = mk_query(X)
  Ox = np.random.randint(0, L)
  Oy = np.random.randint(0, L)
  O = (Ox, Oy)
  if np.random.random() < RAND_HIT:
    ships = get_all_ship_coords(X)
    O = ships[np.random.randint(0, len(ships))]
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
  # each element of shape [batch x ...]
  s1_x = []
  s1_y = []
  s1_o = []
  s2_x = []
  s2_y = []
  s2_o = []
  
  obs_x = [[] for i in range(OBS_SIZE)]
  obs_y = [[] for i in range(OBS_SIZE)]
  obs_tfs = [[] for i in range(OBS_SIZE)]
  new_ob_x = []
  new_ob_y = []
  new_ob_tf = []

  for bb in range(n_batch):
    # generate a hidden variable X
    ships = gen_X()

    # encode the ships in
    ship1 = ships[0]
    ship2 = ships[1]
    # upper left coordinate and a bit for orientation
    ship1_c, ship1_o = ship1[0], ship1[2]
    ship2_c, ship2_o = ship2[0], ship2[2]
    _s1_x, _s1_y = vectorize(ship1_c)
    _s2_x, _s2_y = vectorize(ship2_c)
    s1_x.append(_s1_x)
    s1_y.append(_s1_y)
    s1_o.append(ship1_o)
    s2_x.append(_s2_x)
    s2_y.append(_s2_y)
    s2_o.append(ship2_o)

    # generate new observation
    _new_ob_coord, _new_ob_lab = gen_O(ships)
    _new_ob_x, _new_ob_y = vectorize(_new_ob_coord)
    new_ob_x.append(_new_ob_x)
    new_ob_y.append(_new_ob_y)
    new_ob_tf.append(_new_ob_lab)

    # generate observations for this hidden variable x
    for ob_idx in range(OBS_SIZE):
      _ob_coord, _ob_lab = gen_O(ships)
      _ob_x, _ob_y = vectorize(_ob_coord)
      obs_x[ob_idx].append(_ob_x)
      obs_y[ob_idx].append(_ob_y)
      obs_tfs[ob_idx].append(_ob_lab)

  return  np.array(s1_x, np.float32),\
          np.array(s1_y, np.float32),\
          np.array(s1_o, np.float32),\
          np.array(s2_x, np.float32),\
          np.array(s2_y, np.float32),\
          np.array(s2_o, np.float32),\
          np.array(obs_x, np.float32),\
          np.array(obs_y, np.float32),\
          np.array(obs_tfs, np.float32),\
          np.array(new_ob_x, np.float32),\
          np.array(new_ob_y, np.float32),\
          np.array(new_ob_tf, np.float32)

