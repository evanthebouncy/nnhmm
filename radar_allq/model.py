import tensorflow as tf
import numpy as np
from data import *

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# set up placeholders
ph_x_x = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_x_x")
ph_x_y = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_x_y")
ph_obs_x = [tf.placeholder(tf.float32, [N_BATCH, L], 
            name="ph_ob_x"+str(i)) for i in range(OBS_SIZE)]
ph_obs_y = [tf.placeholder(tf.float32, [N_BATCH, L],
            name="ph_ob_y"+str(j)) for j in range(OBS_SIZE)]
ph_obs_tf = [tf.placeholder(tf.float32, [N_BATCH, 2],
            name="ph_ob_tf"+str(k)) for k in range(OBS_SIZE)]
ph_new_obs = [tf.placeholder(tf.float32, [N_BATCH, L+L], name="ph_new_obs"+str(jjj))
             for jjj in range(L*L)]
ph_new_ob_tfs = [tf.placeholder(tf.float32, [N_BATCH,2], name="ph_new_ob_tfs"+str(jjj))
                 for jjj in range(L*L)]

def gen_feed_dict(x_x, x_y, obs_x, obs_y, obs_tf, 
                  new_obs, new_ob_tfs):
  ret = {}
  for a, b in zip(ph_obs_x, obs_x):
    ret[a] = b
  for a, b in zip(ph_obs_y, obs_y):
    ret[a] = b
  for a, b in zip(ph_obs_tf, obs_tf):
    ret[a] = b
  for a, b in zip(ph_new_obs, new_obs):
    ret[a] = b
  for a, b in zip(ph_new_ob_tfs, new_ob_tfs):
    ret[a] = b

  ret[ph_x_x] = x_x
  ret[ph_x_y] = x_y
  return ret

# some constants
n_hidden = 200
n_pred_hidden = 200

# a list of variables for different tasks
VAR_inv = []
VAR_pred = []

# --------------------------------------------------------------------- initial hidden h(X)
# set up weights for input outputs!
hidden = tf.Variable(tf.truncated_normal([1, n_hidden], stddev=0.1))
print "initial hidden dim ", show_dim(hidden)
hidden_tile = tf.tile(hidden, [N_BATCH, 1])
print "tiled hidden dim ", show_dim(hidden_tile)

# ------------------------------------------------------------------ convolve in the observations

# initialize some weights
W_ob_enc = weight_variable([n_hidden + L + L + 2, n_hidden])
b_ob_enc = bias_variable([n_hidden])

VAR_inv += [W_ob_enc, b_ob_enc]
VAR_pred += [W_ob_enc, b_ob_enc]

hidden_rollin = hidden_tile
hiddens = [hidden_tile]
for i in range(OBS_SIZE):
  print "volvoing input ", i
  ob_xx = ph_obs_x[i]
  ob_yy = ph_obs_y[i]
  ob_tf = ph_obs_tf[i]
  print "input dim ", show_dim(ob_xx), show_dim(ob_yy)
  # concatenate the hidden with the input into a joint channal
  hidden_cat_ob = tf.concat(1, [hidden_rollin, ob_xx, ob_yy, ob_tf])
  print "concat dim of hidden and ob ", show_dim(hidden_cat_ob)
  # convolve them into the new hidden representation 
  hidden_rollin = tf.nn.relu(tf.matmul(hidden_cat_ob, W_ob_enc) + b_ob_enc)

  hiddens.append(hidden_rollin)
  print "rollin dim after takin in inputs ", show_dim(hidden_rollin)

print "hidden shape ", show_dim(hiddens)

# -------------------------------------------------------------------- invert to predict hidden X
W_inv_x = weight_variable([n_hidden,L])
b_inv_x = bias_variable([L])
W_inv_y = weight_variable([n_hidden,L])
b_inv_y = bias_variable([L])

VAR_inv += [W_inv_x, b_inv_x, W_inv_y, b_inv_y]

epsilon1 = tf.constant(1e-10, shape=[N_BATCH, L])
x_invs = [tf.nn.softmax(tf.matmul(volvoo, W_inv_x) + b_inv_x)+epsilon1 for volvoo in hiddens]
y_invs = [tf.nn.softmax(tf.matmul(volvoo, W_inv_y) + b_inv_y)+epsilon1 for volvoo in hiddens]
print "invs shapes ", show_dim(x_invs), show_dim(y_invs)

# compute costs
inv_costs_x = [-tf.reduce_sum(ph_x_x * tf.log(x_pred)) for x_pred in x_invs]
inv_costs_y = [-tf.reduce_sum(ph_x_y * tf.log(y_pred)) for y_pred in y_invs]
print "costs shapes ", show_dim(inv_costs_x)
cost_inv = sum(inv_costs_x) + sum(inv_costs_y)

# ----------------------------------------------------------------- answer all the querys
W_query1 = weight_variable([n_hidden + L + L, n_pred_hidden])
b_query1 = bias_variable([n_pred_hidden])
W_query2 = weight_variable([n_pred_hidden, 2])
b_query2 = bias_variable([2])
VAR_pred += [W_query1, b_query1, W_query2, b_query2]

# for each observation step
# create L * L pairs of query answering modules
# then answer a buttload of queries
all_obs_preds = [[] for _ in range(OBS_SIZE + 1)]

# for each prefix of observation hidden stuff do L * L observations
e2 = tf.constant(1e-10, shape=[N_BATCH, 2])

for ob_ijk in range(OBS_SIZE + 1):
  for ijk in range(L*L):
    qry_inn = ph_new_obs[ijk]
    hidden = hiddens[ob_ijk]
    hidden_cat_query = tf.nn.relu(\
      tf.matmul(tf.concat(1, [ph_new_obs[ijk], hidden]),W_query1) + b_query1)
    query_pred = tf.nn.softmax(tf.matmul(hidden_cat_query, W_query2) + b_query2) + e2
    
    all_obs_preds[ob_ijk].append(query_pred)

print "query_preds shape ", show_dim(all_obs_preds)

query_pred_costs = []
for ob_ijk in range(OBS_SIZE + 1):
  for ijk in range(L*L):
    new_ob_tf = ph_new_ob_tfs[ijk]
    op = all_obs_preds[ob_ijk][ijk]
    query_pred_cost = -tf.reduce_sum(new_ob_tf * tf.log(op))
    query_pred_costs.append(query_pred_cost)
print "costs shapes ", show_dim(query_pred_costs)
cost_query_pred = sum(query_pred_costs)

# ------------------------------------------------------------------------ training steps
optimizer = tf.train.RMSPropOptimizer(0.001)
train_inv = optimizer.minimize(cost_inv, var_list = VAR_inv)
train_query_pred = optimizer.minimize(cost_query_pred, var_list = VAR_pred)
# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# ------------------------------------------------------------------------- using the model!

# train the model and save checkpt to a file location
def train_model(save_loc):
  # Launch the graph.
  sess = tf.Session()
  sess.run(init)
  saver = tf.train.Saver()


  for i in range(5000001):
    x_x, x_y, obs_x, obs_y, obs_tfs, new_obs, new_ob_tfs = gen_data()
    feed_dic = gen_feed_dict(x_x, x_y, obs_x, obs_y, obs_tfs, new_obs, new_ob_tfs)

    # train inversion
    cost_inv_pre = sess.run([cost_inv], feed_dict=feed_dic)[0]
    sess.run([train_inv], feed_dict=feed_dic)
    cost_inv_post = sess.run([cost_inv], feed_dict=feed_dic)[0]
    print "train inv ", cost_inv_pre, " ", cost_inv_post, " ", True if cost_inv_post < cost_inv_pre else False
    # train query prediction
    cost_query_pred_pre = sess.run([cost_query_pred], feed_dict=feed_dic)[0]
    sess.run([train_query_pred], feed_dict=feed_dic)
    cost_query_pred_post = sess.run([cost_query_pred], feed_dict=feed_dic)[0]
    print "train query pred ", cost_query_pred_pre, " ", cost_query_pred_post, " ", True if cost_query_pred_post < cost_query_pred_pre else False

    if i % 100 == 0:
      print "for inversion"
      ran_x_invs = sess.run(x_invs, feed_dict=feed_dic)
      ran_y_invs = sess.run(y_invs, feed_dict=feed_dic)
      print "inverted "
      print ran_x_invs[9][0], np.argmax(ran_x_invs[9][0])
      print ran_y_invs[9][0], np.argmax(ran_y_invs[9][0]) 
      print "true " 
      print x_x[0], np.argmax(x_x[0])
      print x_y[0], np.argmax(x_y[0])

      ran_all_obs_preds = sess.run(all_obs_preds, feed_dict=feed_dic)
      ran_all_obs_preds0 = []
      for id_ob in range(OBS_SIZE):
        for id_LL in range(L*L):
          ran_all_obs_preds0.append(ran_all_obs_preds[id_ob][id_LL][0])

      print "query"
      print ran_all_obs_preds0

      save_path = saver.save(sess, save_loc)
      print("Model saved in file: %s" % save_path)


# load the model and give back a session
def load_model(saved_loc):
  saver = tf.train.Saver()
  sess = tf.Session()
  saver.restore(sess, saved_loc)
  print("Model restored.")
  return sess

