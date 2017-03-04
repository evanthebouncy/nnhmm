from model import *
from draw import *
# ------------- helpers --------------
# some on-policy data generation
def full_output_to_trace(full_out, Img, S):
  obs = [x[0] for x in full_out[1:]]
  return Trace(Img, S, obs)

# start the training process some arguments
restart = True

invnet = Invnet("inv")
sess = tf.Session()

exp = Experience(500)
for i in range(500):
  exp.add(gen_rand_trace())

if restart:
  sess.run(invnet.init)
else:
  impnet.load_model(sess, "model_invert.ckpt")

for i in xrange(1000000):
  print i
  # train it on a sample of expeirence
  lab_batch = []
  img_batch = []
  for _ in range(N_BATCH):
    tr = exp.sample()
    lab_batch.append(tr.S)
    img_batch.append(tr.Img)
  invnet.train(sess, inv_data_from_label_data(lab_batch, img_batch))

  if i % 1000 == 0:
    print "testing it a bit . . ."
    rand_tr = gen_rand_trace(test=True)
    print rand_tr.S, np.argmax(rand_tr.S)
    invv = invnet.invert(sess, rand_tr.Os)
    print invv, np.argmax(invv)
    draw_obs(rand_tr.Os, "drawings/inv_obs.png")
    invnet.save(sess, "model_invert.ckpt")
    

