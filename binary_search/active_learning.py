from model import *

# ------------------------------------------------------------------ helpers and mgmt
def get_feed_dic_obs(obs):
  # needing to create all the nessisary feeds
  obs_x = []
  obs_tf = []
  
  for _ in range(OBS_SIZE):
    obs_x.append(np.zeros([N_BATCH,L]))
    obs_tf.append(np.zeros([N_BATCH,2]))

  num_obs = len(obs)
  for ob_idx in range(num_obs):
    ob_coord, ob_lab = obs[ob_idx]
    ob_x = vectorize(ob_coord, L)
    obs_x[ob_idx] = np.tile(ob_x, [50,1])
    obs_tf[ob_idx] = np.tile(ob_lab, [50,1])

  feed_dic = dict(zip(ph_obs_x + ph_obs_tf, 
                      obs_x + obs_tf))
  return feed_dic

def get_inv(sess, obs):
  num_obs = len(obs)
  return get_inv_tr(sess, obs)[num_obs]

def get_inv_tr(sess, obs):
  num_obs = len(obs)
  feed_dic = get_feed_dic_obs(obs)
  x_invss = [np.argmax(x[0]) for x in sess.run(x_invs, feed_dict = feed_dic)]
  return x_invss

def get_most_confuse(sess, obs):
  feed_dic = get_feed_dic_obs(obs)
  key_ob = len(obs)

  all_querys = []
  for i in range(L):
    all_querys.append(i)

  most_conf = (1.0, None)

  for q in all_querys:
    q_x = vectorize(q, L)
    feed_dic[ph_new_ob_x] = np.tile(q_x, [N_BATCH,1])
    pred_tf = sess.run(query_preds, feed_dict=feed_dic)[key_ob][0]
    most_conf = min(most_conf, (abs(pred_tf[0] - pred_tf[1]), q)) 

  return most_conf[1]

def get_random_inv(sess, query):
  ob_pts = [np.random.randint(0, L) for _ in range(OBS_SIZE)]
  obs = [(op, query(op)) for op in ob_pts]
  return zip([None] + obs, get_inv_tr(sess, obs))

def get_active_inv(sess, query):
  obs = []

  for i in range(OBS_SIZE):
    try_inv = get_inv(sess, obs)
    most_conf = get_most_confuse(sess, obs)
    print "chosen observation ", most_conf
    obs.append((most_conf, query(most_conf)))

  return zip([None] + obs, get_inv_tr(sess, obs))
  
