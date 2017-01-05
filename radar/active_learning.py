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
  feed_dic = get_feed_dic_obs(obs)
  return sess.run(x_invs, feed_dict = feed_dic)[num_obs-1][0],\
         sess.run(y_invs, feed_dict = feed_dic)[num_obs-1][0]

sess = load_model("model.ckpt")

print get_inv(sess, [ ((1,1),[1.0, 0.0]),
                     ((10,10),[0.0,1.0]) ])

hid_x = gen_X()
obs = [gen_O(hid_x) for i in range(10)]

print hid_x
res_x, res_y = get_inv(sess, obs)
print np.argmax(res_x), np.argmax(res_y)

def get_most_confuse(sess, obs):
  feed_dic = get_feed_dic_obs(obs)
  key_ob = len(obs)-1

  all_querys = []
  for i in range(L):
    for j in range(L):
      all_querys.append((i,j))

  for q in all_querys:
    q_x, q_y = vectorize(q)
    print q
    feed_dic[ph_new_ob_x] = np.tile(q_x, [N_BATCH,1])
    feed_dic[ph_new_ob_y] = np.tile(q_y, [N_BATCH,1])
    pred_tf = sess.run(query_preds, feed_dict=feed_dic)[key_ob][0]

    print pred_tf

print get_most_confuse(sess, obs)

