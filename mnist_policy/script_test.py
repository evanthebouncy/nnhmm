from model import *
from draw import *
# ------------- helpers --------------
# some on-policy data generation
def full_output_to_trace(full_out, Img, S):
  obs = [x[0] for x in full_out[1:]]
  return Trace(Img, S, obs)

impnet = Implynet()
invnet = Invnet("inv")
sess = tf.Session()
impnet.load_model(sess, "model_imply.ckpt")
invnet.load_model(sess, "model_invert.ckpt")

# cor_rand = 0.0
# cor_active = 0.0
cor_rand = 0.0
cor_active = 0.0
for i in range(1,10000):
  img, _x = get_img_class(test=True)
  query = mk_query(img)
  rand_inv = impnet.get_random_inv(sess, query)
  acti_inv = impnet.get_active_inv(sess, query)

  rand_obs = [xx[0] for xx in rand_inv[1:]]
  acti_obs = [xx[0] for xx in acti_inv[1:]]

  rand_answer = invnet.invert(sess, rand_obs)
  acti_answer = invnet.invert(sess, acti_obs)

  if np.argmax(rand_answer) == np.argmax(_x):
    cor_rand += 1
  if np.argmax(acti_answer) == np.argmax(_x):
    cor_active += 1


  print "avg rand ", cor_rand, " ", cor_rand / i , " ", i
  print "avg acti ", cor_active, " ", cor_active / i , " ", i


