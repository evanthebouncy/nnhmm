from model import *
from draw import *
# ------------- helpers --------------
# some on-policy data generation
def full_output_to_trace(full_out, Img, S):
  obs = [x[0] for x in full_out[1:]]
  return Trace(Img, S, obs)

# start the training process some arguments
restart = False
train_unsupervised = False


exp = Experience(2000)
impnet = Implynet()
sess = tf.Session()


if train_unsupervised:
  # add some initial stuff
  init_trs = [gen_rand_trace() for _ in range(N_BATCH)]
  for blah in init_trs:
    exp.add(blah)

  if restart:
    sess.run(impnet.init)
  else:
    impnet.load_model(sess, "model.ckpt")

  epi = 1.0
  for i in xrange(1000000):
    epi = 0.9 ** mnist.train.epochs_completed
    print i, " epsilon ", epi, " epoch num", mnist.train.epochs_completed
    # train it on a sample of expeirence
    impnet.train(sess, data_from_exp(exp))

    # add a random sample ?
    # exp.add(gen_rand_trace())
    # add an active sample
    imga, xa = get_img_class()
    qry = mk_query(imga)
    act_inv = impnet.get_active_inv(sess, qry, epi)
    act_tr = full_output_to_trace(act_inv, imga, xa)
    exp.add(act_tr)

    if i % 100 == 0:
      print "Drawing some images . . ."
      ran_inv = impnet.get_random_inv(sess, qry)
      for i in range(OBS_SIZE+1):
        draw_all_preds(ran_inv[i][1], "drawings/rand_inv{0}.png".format(i))
        draw_all_preds(act_inv[i][1], "drawings/acti_inv{0}.png".format(i))
      draw_trace(ran_inv, "drawings/rand_inv_tr.png")
      draw_trace(act_inv, "drawings/acti_inv_tr.png")
      draw(np.reshape(imga, [L,L,1]), "drawings/truth.png")
      impnet.save(sess)

if not train_unsupervised:
  impnet.load_model(sess, "model.ckpt")

  # add some initial stuff
  init_trs = [gen_rand_trace() for _ in range(10 * N_BATCH)]
  for blah in init_trs:
    exp.add(blah)

  for i in xrange(1000000):
    print i
    impnet.train(sess, data_from_exp(exp), train_inv=True)

    # add an active random sample from these 50 images only
    trace = exp.sample()
    imga, xa = trace.Img, trace.S
    qry = mk_query(imga)

    # act_inv = impnet.get_active_inv(sess, qry)
    # act_tr = full_output_to_trace(act_inv, imga, xa)
    # exp.add(act_tr)

    ran_inv = impnet.get_random_inv(sess, qry)
    ran_tr = full_output_to_trace(ran_inv, imga, xa)
    exp.add(ran_tr)

    if i % 100 == 0:
      # print "Drawing some images . . ."
      # ran_inv = impnet.get_random_inv(sess, qry)
      # for i in range(OBS_SIZE+1):
      #   draw_all_preds(ran_inv[i][1], "drawings/rand_inv{0}.png".format(i))
      #   draw_all_preds(act_inv[i][1], "drawings/acti_inv{0}.png".format(i))
      # draw_trace(ran_inv, "drawings/rand_inv_tr.png")
      # draw_trace(act_inv, "drawings/acti_inv_tr.png")
      # draw(np.reshape(imga, [L,L,1]), "drawings/truth.png")
      # print act_inv[-1][2][0], act_inv[-1][2][1] 
      # print xa, np.argmax(xa)
      impnet.save(sess)


