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
ph_obs_x = [tf.placeholder(tf.float32, [N_BATCH, L], 
            name="ph_ob_x"+str(i)) for i in range(OBS_SIZE)]
ph_obs_y = [tf.placeholder(tf.float32, [N_BATCH, L],
            name="ph_ob_y"+str(j)) for j in range(OBS_SIZE)]
ph_obs_tf = [tf.placeholder(tf.float32, [N_BATCH, 2],
            name="ph_ob_tf"+str(k)) for k in range(OBS_SIZE)]
ph_new_ob_x = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_new_ob_x")
ph_new_ob_y = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_new_ob_y")
ph_new_ob_tf = tf.placeholder(tf.float32, [N_BATCH,2], name="ph_new_ob_tf")

def gen_feed_dict(obs_x, obs_y, obs_tf, 
                  new_ob_x, new_ob_y, new_ob_tf):
  ret = {}
  for a, b in zip(ph_obs_x, obs_x):
    ret[a] = b
  for a, b in zip(ph_obs_y, obs_y):
    ret[a] = b
  for a, b in zip(ph_obs_tf, obs_tf):
    ret[a] = b

  ret[ph_new_ob_x] = new_ob_x
  ret[ph_new_ob_y] = new_ob_y
  ret[ph_new_ob_tf] = new_ob_tf
  return ret

# some constants
n_hidden = 400
n_pred_hidden = 400

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
hiddens = [hidden_rollin]
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

# ----------------------------------------------------------------- answer the query
W_query1 = weight_variable([n_hidden + L + L, n_pred_hidden])

b_query1 = bias_variable([n_pred_hidden])
W_query2 = weight_variable([n_pred_hidden, 2])
b_query2 = bias_variable([2])
VAR_pred += [W_query1, b_query1, W_query2, b_query2]

hidden_cat_query = [tf.nn.relu(\
  tf.matmul(tf.concat(1, [ph_new_ob_x, ph_new_ob_y, hidden]),W_query1) + b_query1)\
  for hidden in hiddens]

print "hidden_cat_query shape ", show_dim(hidden_cat_query)
e2 = tf.constant(1e-10, shape=[N_BATCH, 2])
query_preds = [tf.nn.softmax(tf.matmul(hcq, W_query2) + b_query2)+e2 for hcq in hidden_cat_query]
print "query_preds shape ", show_dim(query_preds)

query_pred_costs = [-tf.reduce_sum(ph_new_ob_tf * tf.log(op)) for op in query_preds]
print "costs shapes ", show_dim(query_pred_costs)
cost_query_pred = sum(query_pred_costs)

# ------------------------------------------------------------------------ training steps
optimizer = tf.train.RMSPropOptimizer(0.001)
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
    obs_x, obs_y, obs_tfs, new_ob_x, new_ob_y, new_ob_tf = gen_data()
    feed_dic = gen_feed_dict(obs_x, obs_y, obs_tfs, new_ob_x, new_ob_y, new_ob_tf)

    # train query prediction
    cost_query_pred_pre = sess.run([cost_query_pred], feed_dict=feed_dic)[0]
    sess.run([train_query_pred], feed_dict=feed_dic)
    cost_query_pred_post = sess.run([cost_query_pred], feed_dict=feed_dic)[0]
    print "train query pred ", cost_query_pred_pre, " ", cost_query_pred_post, " ", True if cost_query_pred_post < cost_query_pred_pre else False

    if i % 100 == 0:
      print "for query prediction"
      print "query loc "
      print new_ob_x[0]
      print new_ob_y[0]
      print "predicted <===> true"
      ran_predss = sess.run(query_preds, feed_dict=feed_dic)[9]
      total_cor = 0.0
      for haha in zip(ran_predss, new_ob_tf):
        print haha
        if np.argmax(haha[0]) == np.argmax(haha[1]):
          total_cor += 1
      cor_ratio = total_cor / len(ran_predss)
      print "total correct ", cor_ratio
      
      # something special!
      global RAND_HIT
      NEW_RAND_HIT = 1.0 - cor_ratio
      RAND_HIT = RAND_HIT * 0.9 + NEW_RAND_HIT * 0.1
      print "rand hit now ", RAND_HIT


      save_path = saver.save(sess, save_loc)
      print("Model saved in file: %s" % save_path)


# load the model and give back a session
def load_model(saved_loc):
  saver = tf.train.Saver()
  sess = tf.Session()
  saver.restore(sess, saved_loc)
  print("Model restored.")
  return sess

