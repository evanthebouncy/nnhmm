from active_learning import *
from draw import *

# OBS_SIZE = 20

np.set_printoptions(precision=3)

# get a hidden number! 
for i in range(np.random.randint(0, 50)):
  img, _x = get_img_class()

print img
print _x, np.argmax(_x)

draw(np.reshape(img, [L,L,1]), "drawings/truth.png")

sess = load_model("model.ckpt")

query = mk_query(img)

# all_p_f = get_all_preds_fast(sess, [])
# all_p = get_all_preds(sess, [])
# 
# draw_all_preds(all_p_f[0], "fast.png")
# draw_all_preds(all_p[0], "slow.png")

rand_inv = get_random_inv(sess, query, OBS_SIZE=OBS_SIZE)
print "done with random inversion "
for i in range(OBS_SIZE+1):
  draw_all_preds(rand_inv[i][1], "drawings/rand_inv{0}.png".format(i))

for x in rand_inv:
  print x[0], x[2]

print "doing active inv"
acti_inv = get_active_inv(sess, query, OBS_SIZE=OBS_SIZE)
print "done with active inv"

# draw_ship(hid_x, "drawings/orig.png")
for i in range(OBS_SIZE+1):
  draw_all_preds(acti_inv[i][1], "drawings/acti_inv{0}.png".format(i))

for x in acti_inv:
  print x[0], x[2]

draw_trace(rand_inv, "drawings/rand_inv_tr.png")
draw_trace(acti_inv, "drawings/acti_inv_tr.png")

'''
# cor_rand = 0.0
# cor_active = 0.0
cor_rand = 3204.0
cor_active = 4536.0
for i in range(1,10000):
  
  img, _x = get_img_class(test=True)
  query = mk_query(img)

  if i < 7610:
    continue

  rand_inv = get_random_inv(sess, query, OBS_SIZE=OBS_SIZE)
  acti_inv = get_active_inv(sess, query, OBS_SIZE=OBS_SIZE)

  if rand_inv[-1][2][1] == np.argmax(_x):
    cor_rand += 1

  if acti_inv[-1][2][1] == np.argmax(_x):
    cor_active += 1

  print "avg rand ", cor_rand, " ", cor_rand / i , " ", i
  print "avg acti ", cor_active, " ", cor_active / i , " ", i
'''
