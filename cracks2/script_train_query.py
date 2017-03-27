from model import *
from draw import *
# ------------- helpers --------------
# some on-policy data generation
def full_output_to_trace(full_out, Img, S):
  obs = [x[0] for x in full_out[1:]]
  return Trace(Img, S, obs)

# start the training process some arguments
restart = True

exp = Experience(2000)
impnet = Implynet("imp")
sess = tf.Session()

# add some initial stuff
init_trs = [gen_rand_trace() for _ in range(N_BATCH)]
for blah in init_trs:
  exp.add(blah)

if restart:
  sess.run(impnet.init)
else:
  impnet.load_model(sess, "model_imply.ckpt")

epi = 1.0
for i in xrange(1000000):
  # epi = 0.9 ** mnist.train.epochs_completed
  epi = 0.9 ** (i / 1000)
  print i, " epsilon ", epi, " epoch num", mnist.train.epochs_completed
  # train it on a sample of expeirence
  impnet.train(sess, data_from_exp(exp, epi))

  # add a random sample ?
  # exp.add(gen_rand_trace())
  # add an active sample
  imga, xa = get_img_class()
  qry = mk_query(xa)
  act_inv_obs = impnet.get_active_trace(sess, qry, epi)
  # act_tr = full_output_to_trace(act_inv, imga, xa)
  act_tr = Trace(imga, xa, act_inv_obs)
  exp.add(act_tr)

  if i % 100 == 0:
    print "Drawing some images . . ."
    ran_inv_obs = impnet.get_active_trace(sess, qry, 1.0)
    act_inv_obs = impnet.get_active_trace(sess, qry, 0.0)
    acti_all_preds = [impnet.get_all_preds_fast(sess, act_inv_obs[:i]) for i in range(OBS_SIZE+1)]
    rand_all_preds = [impnet.get_all_preds_fast(sess, ran_inv_obs[:i]) for i in range(OBS_SIZE+1)]

    for i in range(OBS_SIZE+1):
      draw_all_preds(rand_all_preds[i], "drawings/rand_inv{0}.png".format(i))
      draw_all_preds(acti_all_preds[i], "drawings/acti_inv{0}.png".format(i))
    draw_obs_trace(ran_inv_obs, "drawings/rand_inv_tr.png")
    draw_obs_trace(act_inv_obs, "drawings/acti_inv_tr.png")
    draw(np.reshape(imga, [L,L,1]), "drawings/truth.png")
    impnet.save(sess, "model_imply.ckpt")

