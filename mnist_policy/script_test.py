from model import *
from draw import *
# ------------- helpers --------------
# some on-policy data generation
def full_output_to_trace(full_out, Img, S):
  obs = [x[0] for x in full_out[1:]]
  return Trace(Img, S, obs)

impnet = Implynet()
sess = tf.Session()
impnet.load_model(sess, "model.ckpt")

# cor_rand = 0.0
# cor_active = 0.0
cor_rand = 0.0
cor_active = 0.0
for i in range(1,10000):
  img, _x = get_img_class(test=True)
  query = mk_query(img)
  rand_inv = impnet.get_random_inv(sess, query)
  acti_inv = impnet.get_active_inv(sess, query)

  if rand_inv[-1][2][1] == np.argmax(_x):
    cor_rand += 1

  if acti_inv[-1][2][1] == np.argmax(_x):
    cor_active += 1

  print "avg rand ", cor_rand, " ", cor_rand / i , " ", i
  print "avg acti ", cor_active, " ", cor_active / i , " ", i


