from model import *

# ------------------------------------------------------------------ helpers and mgmt
def get_feed_dic_obs(obs):
  # needing to create all the nessisary feeds
  obs_x = []
  obs_y = []
  obs_tf = []
  
  for _ in range(OBS_SIZE):
    obs_x.append(np.zeros([N_BATCH,L]))
    obs_y.append(np.zeros([N_BATCH,L]))
    obs_tf.append(np.zeros([N_BATCH,2]))

  num_obs = len(obs)
  for ob_idx in range(num_obs):
    ob_coord, ob_lab = obs[ob_idx]
    ob_x, ob_y = vectorize(ob_coord)
    obs_x[ob_idx] = np.tile(ob_x, [50,1])
    obs_y[ob_idx] = np.tile(ob_y, [50,1])
    obs_tf[ob_idx] = np.tile(ob_lab, [50,1])

  feed_dic = dict(zip(ph_obs_x + ph_obs_y + ph_obs_tf, 
                      obs_x + obs_y + obs_tf))
  return feed_dic

def get_inv(sess, obs):
  num_obs = len(obs)
  return get_inv_tr(sess, obs)[num_obs]

def get_inv_tr(sess, obs):
  ret_min =  sess.run(inv_a_mins, feed_dict=get_feed_dic_obs(obs))
  ret_max =  sess.run(inv_a_maxs, feed_dict=get_feed_dic_obs(obs))

  ret_min = [x[0] for x in ret_min]
  ret_max = [x[0] for x in ret_max]
  # ret_min = [np.argmax(x[0]) for x in ret_min]
  # ret_max = [np.argmax(x[0]) for x in ret_max]
  return zip(ret_min, ret_max)

def get_all_query_score(sess, obs):
  feed_dic = get_feed_dic_obs(obs)
  all_querys = []
  for i in range(L):
    for j in range(L):
      all_querys.append((i,j))

  qry_idx = len(obs)
  ret = dict()

  for q in all_querys:
    q_x, q_y = vectorize(q)
    feed_dic[ph_new_ob_x] = np.tile(q_x, [N_BATCH,1])
    feed_dic[ph_new_ob_y] = np.tile(q_y, [N_BATCH,1])
    qry_score = sess.run([conditional_ent[qry_idx]], feed_dict=feed_dic)
    ret[q] = qry_score[0][0]

  return ret

def get_best_query(sess, obs):
  all_qry_score = get_all_query_score(sess, obs)
  all_qry_score = [(x[1],x[0]) for x in all_qry_score.items()]
  return min(all_qry_score)

def get_random_inv(sess, query):
  ob_pts = [(np.random.randint(0, L), np.random.randint(0,L)) for _ in range(OBS_SIZE)]
  obs = [(op, query(op)) for op in ob_pts]
  return zip([None] + obs, get_all_preds(sess, obs))

def get_random_inv_direct(sess, query):
  ob_pts = [(np.random.randint(0, L), np.random.randint(0,L)) for _ in range(OBS_SIZE)]
  obs = [(op, query(op)) for op in ob_pts]
  ret =  sess.run([inv_a_mins[OBS_SIZE],
                   inv_a_maxs[OBS_SIZE]], feed_dict=get_feed_dic_obs(obs))
  ret_score = sess.run(query_score, feed_dict=get_feed_dic_obs(obs))
  return [x[0] for x in ret] + [[x[0] for x in ret_score]]

def get_active_inv_direct(sess, query):
  obs = []
  for i in range(OBS_SIZE):
    most_conf = get_most_confuse(sess, obs)
    obs.append((most_conf, query(most_conf)))
  ret =  sess.run([inv_s1_xs[OBS_SIZE],
                   inv_s1_ys[OBS_SIZE],
                   inv_s1_os[OBS_SIZE],
                   inv_s2_xs[OBS_SIZE],
                   inv_s2_ys[OBS_SIZE],
                   inv_s2_os[OBS_SIZE]], feed_dict=get_feed_dic_obs(obs))
  return [x[0] for x in ret]
  

def get_active_inv(sess, query):
  obs = []

  for i in range(OBS_SIZE):
    best_qry = get_best_query(sess, obs)[1]
    obs.append((best_qry, query(best_qry)))

  return zip([None] + obs, get_inv_tr(sess, obs))
  
