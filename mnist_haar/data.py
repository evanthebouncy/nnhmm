import numpy as np
import pywt
from scipy.misc import imresize
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X_L = 10
L = 14
N_BATCH = 50
OBS_SIZE = 30


# ---------------------------- helpers
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

# -------------------------------------- making the datas

# assume X is already a 2D matrix
def mk_query(X):
  avg = np.median(X)
  X = X + avg
  def query(O):
    Ox, Oy = O
    if X[Ox][Oy] > 0.0:
      return [1.0, 0.0]
    else:
      return [0.0, 1.0]
  return query

def sample_coord():
  Ox, Oy = np.random.randint(0,L), np.random.randint(0,L)
  if 0 <= Ox < L:
    if 0 <= Oy < L:
      return Ox, Oy
  return sample_coord()

def gen_O(X):
  query = mk_query(X)
  Ox, Oy = sample_coord()
  O = (Ox, Oy)
  return O, query(O) 

def get_img_class():
  img, _x = mnist.train.next_batch(1)
  img = np.reshape(img[0], [28, 28])
  img = imresize(img, (L,L)) / 255.0
  A,(B,C,D) = pywt.dwt2(img, 'haar')
  img = np.reshape(np.array([A,B,C,D]), [L, L])
  return img, _x

def gen_data():
  x = []
  
  obs_x = [[] for i in range(OBS_SIZE)]
  obs_y = [[] for i in range(OBS_SIZE)]
  obs_tfs = [[] for i in range(OBS_SIZE)]
  new_ob_x = []
  new_ob_y = []
  new_ob_tf = []

  imgs = []

  for bb in range(N_BATCH):
    # generate a hidden variable X
    # get a single thing out
    img, _x = get_img_class()
    imgs.append(img)

    # add to x
    x.append(_x[0])
    # generate new observation
    _new_ob_coord, _new_ob_lab = gen_O(img)
    _new_ob_x, _new_ob_y = vectorize(_new_ob_coord)
    new_ob_x.append(_new_ob_x)
    new_ob_y.append(_new_ob_y)
    new_ob_tf.append(_new_ob_lab)

    # generate observations for this hidden variable x
    for ob_idx in range(OBS_SIZE):
      _ob_coord, _ob_lab = gen_O(img)
      _ob_x, _ob_y = vectorize(_ob_coord)
      obs_x[ob_idx].append(_ob_x)
      obs_y[ob_idx].append(_ob_y)
      obs_tfs[ob_idx].append(_ob_lab)

  return  np.array(x, np.float32),\
          np.array(obs_x, np.float32),\
          np.array(obs_y, np.float32),\
          np.array(obs_tfs, np.float32),\
          np.array(new_ob_x, np.float32),\
          np.array(new_ob_y, np.float32),\
          np.array(new_ob_tf, np.float32), imgs
