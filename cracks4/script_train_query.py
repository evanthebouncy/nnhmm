from model import *
from draw import *

# start the training process some arguments
restart = False

impnet = Implynet("imp")
sess = tf.Session()

if restart:
  sess.run(impnet.init)
else:
  impnet.load_model(sess, "model_imply.ckpt")

epi = 1.0
for i in xrange(1000000):
  # epi = 0.9 ** mnist.train.epochs_completed
  epi = max(0.1, 0.9 ** (i / 50))
  if epi == 0.1:
    epi = np.random.random() / 10 
  print i, " ", epi
  # train it on a sample of expeirence
  impnet.train(sess, rand_data(epi))

  if i % 20 == 0:
    partial, full = rand_data(epi)
    predzz = sess.run(impnet.query_preds, impnet.gen_feed_dict(partial, full))
    predzz0 = np.array([x[0] for x in predzz])
    print show_dim(predzz0)
    predzz0 = np.reshape(predzz0, [L,L,2])
    draw_allob(predzz0, "drawings/pred_ob.png")
    draw_allob(full[0], "drawings/orig_ob.png")
    draw_allob(partial[0], "drawings/partial_ob.png")
    impnet.save(sess)

