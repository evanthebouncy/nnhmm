from model import *
RAND_MOVE = 0.5

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

# obs is the shame ob_idx x batch_size x [ob_coord, ob_lab]
def get_feed_dic_obs_batch(obs):
  # needing to create all the nessisary feeds
  obs_x = []
  obs_y = []
  obs_tf = []
  
  # create bunch of 0 place holders
  for _ in range(OBS_SIZE):
    obs_x.append(np.zeros([N_BATCH,L]))
    obs_y.append(np.zeros([N_BATCH,L]))
    obs_tf.append(np.zeros([N_BATCH,2]))

  num_obs = len(obs)
  for ob_idx in range(num_obs):
    for bb in range(N_BATCH):
      ob_coord, ob_lab = obs[ob_idx][bb]
      ob_x, ob_y = vectorize(ob_coord)
      obs_x[ob_idx][bb] = ob_x
      obs_y[ob_idx][bb] = ob_y
      obs_tf[ob_idx][bb] = ob_lab

  feed_dic = dict(zip(ph_obs_x + ph_obs_y + ph_obs_tf, 
                      obs_x + obs_y + obs_tf))
  return feed_dic


def make_obs_tr(sess, true_Xs):
  qry_ans = [mk_query(x) for x in true_Xs]
  all_querys = []
  for i in range(L):
    for j in range(L):
      all_querys.append((i,j))
  
  obs_prefix = []
  for ob_idx in range(OBS_SIZE + 1): 
    # get the argmax prefix first
    obs_feeder = get_feed_dic_obs_batch(obs_prefix)

    most_conf_batches = N_BATCH*[(1.0, None)]

    for q in all_querys:
      q_x, q_y = vectorize(q)
      obs_feeder[ph_new_ob_x] = np.tile(q_x, [N_BATCH,1])
      obs_feeder[ph_new_ob_y] = np.tile(q_y, [N_BATCH,1])
      pred_tf = sess.run(query_preds, feed_dict=obs_feeder)[ob_idx]
      for bb in range(N_BATCH):
        bb_tf = pred_tf[bb]
        most_conf_batches[bb] = min(most_conf_batches[bb], (abs(bb_tf[0] - bb_tf[1]), q)) 

    # randomly put some of the observations to be random
    for bb in range(N_BATCH):
      toss = np.random.random() < RAND_MOVE
      if toss:
        most_conf_batches[bb] = (None, (np.random.randint(L), np.random.randint(L)))  
#    print most_conf_batches
    # WHATS UP WITH THIS CODE HERE LUL
    qry_anss = [(x[1][1], x[0](x[1][1])) for x in zip(qry_ans, most_conf_batches)]
#    print qry_anss
    obs_prefix.append(qry_anss)

  obs_x = [[] for _ in range(OBS_SIZE)]
  obs_y = [[] for _ in range(OBS_SIZE)]
  obs_tfs = [[] for _ in range(OBS_SIZE)]

  # cut away the last observation cuz it will never get queried within the bound
  obs_prefix = obs_prefix[:-1]

  for ob_idx, ob_labs in enumerate(obs_prefix):
    for ob_lab in ob_labs:
      qryyy, lab = ob_lab
      q_x, q_y = vectorize(qryyy)
      obs_x[ob_idx].append(q_x)
      obs_y[ob_idx].append(q_y)
      obs_tfs[ob_idx].append(lab)

  return obs_x, obs_y, obs_tfs, obs_prefix

# given a session, return an observation trace that's partially random and
# partially active learned
# needs to return these batches:
# x_x, x_y, obs_x, obs_y, obs_tf, new_ob_x, new_ob_y, new_ob_tf
def get_active_train_batch(sess):
  # create a n_batch number of 
  x_x = []
  x_y = []
  new_ob_x = []
  new_ob_y = []
  new_ob_tf = []

  true_Xs = []

  for bb in range(N_BATCH):
    # generate a hidden variable X
    x_coord = gen_X()
    true_Xs.append(x_coord)
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
  # half are the arg-max confusion, other half is randomly generated
  obs_x, obs_y, obs_tfs, _ = make_obs_tr(sess, true_Xs)

  return  np.array(x_x, np.float32),\
          np.array(x_y, np.float32),\
          np.array(obs_x, np.float32),\
          np.array(obs_y, np.float32),\
          np.array(obs_tfs, np.float32),\
          np.array(new_ob_x, np.float32),\
          np.array(new_ob_y, np.float32),\
          np.array(new_ob_tf, np.float32)

   

def get_inv(sess, obs):
  num_obs = len(obs)
  return get_inv_tr(sess, obs)[num_obs]

def get_inv_tr(sess, obs):
  num_obs = len(obs)
  feed_dic = get_feed_dic_obs(obs)
  x_invss = [np.argmax(x[0]) for x in sess.run(x_invs, feed_dict = feed_dic)]
  y_invss = [np.argmax(x[0]) for x in sess.run(y_invs, feed_dict = feed_dic)]
  return zip(x_invss, y_invss)

def get_most_confuse(sess, obs):
  feed_dic = get_feed_dic_obs(obs)
  key_ob = len(obs)

  all_querys = []
  for i in range(L):
    for j in range(L):
      all_querys.append((i,j))

  most_conf = (1.0, None)

  for q in all_querys:
    q_x, q_y = vectorize(q)
    feed_dic[ph_new_ob_x] = np.tile(q_x, [N_BATCH,1])
    feed_dic[ph_new_ob_y] = np.tile(q_y, [N_BATCH,1])
    pred_tf = sess.run(query_preds, feed_dict=feed_dic)[key_ob][0]
    most_conf = min(most_conf, (abs(pred_tf[0] - pred_tf[1]), q)) 

  return most_conf[1]

def get_random_inv(sess, query):
  ob_pts = [(np.random.randint(0, L), np.random.randint(0,L)) for _ in range(OBS_SIZE)]
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
  
