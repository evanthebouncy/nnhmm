import numpy as np
from scipy.misc import imresize
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X_L = 10
L = 14
N_BATCH = 50
OBS_SIZE = 20

KEEP = 0.6

# ---------------------------- helpers
def black_white(img):
  new_img = np.copy(img)
  img_flat = img.flatten()
  nonzeros = img_flat[np.nonzero(img_flat)]
  sortedd = np.sort(nonzeros)
  idxx = round(len(sortedd) * (1.0 - KEEP))
  thold = sortedd[idxx]

  mask_pos = img >= thold
  mask_neg = img < thold
  new_img[mask_pos] = 1.0
  new_img[mask_neg] = 0.0
  return new_img

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
  def query(O):
    Ox, Oy = O
    if X[Ox][Oy] == 1.0:
      return [1.0, 0.0]
    else:
      return [0.0, 1.0]
  return query

def sample_coord():
  Ox, Oy = np.random.multivariate_normal([L/2,L/2], [[L*0.7, 0.0], [0.0, L*0.7]])
  Ox, Oy = round(Ox), round(Oy)
  if 0 <= Ox < L:
    if 0 <= Oy < L:
      return Ox, Oy
  return sample_coord()

def sample_coord_bias(qq):
  def find_positive(qq):
    C = sample_coord()
    if qq(C) == [1.0, 0.0]:
      return C
    else:
      return find_positive(qq)
  def find_negative(qq):
    C = sample_coord()
    if qq(C) == [0.0, 1.0]:
      return C
    else:
      return find_negative(qq)

  toss = np.random.random() < 0.5
  if toss:
    return find_positive(qq)
  else:
    return find_negative(qq)

def gen_O(X):
  query = mk_query(X)
  Ox, Oy = sample_coord_bias(query)
  O = (Ox, Oy)
  return O, query(O) 

def get_img_class(test=False):
  img, _x = mnist.train.next_batch(1)
  if test:
    img, _x = mnist.test.next_batch(1)

  img = np.reshape(img[0], [2*L,2*L])                                                           
  # rescale the image to 14 x 14
  img = imresize(img, (14,14), interp='nearest') / 255.0                      
  img = imresize(img, (14,14)) / 255.0                      
  img = black_white(img)
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
