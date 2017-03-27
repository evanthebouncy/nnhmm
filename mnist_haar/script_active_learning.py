from active_learning import *
from draw import *

np.set_printoptions(precision=3)

# get a hidden number! 
for i in range(np.random.randint(0, 50)):
  img, _x = get_img_class()

print img
print _x, np.argmax(_x)

draw(np.reshape(img, [L,L,1]), "drawings/truth.png")

# disable the random hit
RAND_HIT = 0.0

sess = load_model("model.ckpt")


query = mk_query(img)
rand_inv = get_random_inv(sess, query) 
print "done with random inversion "
for i in range(OBS_SIZE+1):
  draw_all_preds(rand_inv[i][1], "drawings/rand_inv{0}.png".format(i))

for x in rand_inv:
  print x[0], x[2]

print "doing active inv"
acti_inv = get_active_inv(sess, query) 
print "done with active inv"

# draw_ship(hid_x, "drawings/orig.png")
for i in range(OBS_SIZE+1):
  draw_all_preds(acti_inv[i][1], "drawings/acti_inv{0}.png".format(i))

for x in acti_inv:
  print x[0], x[2]

draw_trace(rand_inv, "drawings/rand_inv_tr.png")
draw_trace(acti_inv, "drawings/acti_inv_tr.png")

assert 0
cor_rand = 0.0
cor_active = 0.0
for i in range(1,1000):
  hid_x = gen_X()
  query = mk_query(hid_x)

  s1, s2 = hid_x
  s1_x_, s1_y_, s1_o_ = s1[0][0], s1[0][1], s1[2]
  s2_x_, s2_y_, s2_o_ = s2[0][0], s2[0][1], s2[2]

  # ------------------------------------------------- rand inversion
  s1_x, s1_y, s1_o, s2_x, s2_y, s2_o =  get_random_inv_direct(sess, query)
  s1_x = np.argmax(s1_x)
  s1_y = np.argmax(s1_y)
  s1_o = np.argmax(s1_o)

  s2_x = np.argmax(s2_x)
  s2_y = np.argmax(s2_y)
  #s2_o = np.argmax(s2_o)

  if s1_x == s1_x_ and s1_y == s1_y_ and s1_o == np.argmax(s1_o_)\
     and s2_x == s2_x_ and s2_y == s2_y_:
    cor_rand += 1

  # -------------------------------------------------- active inversion
  s1_x, s1_y, s1_o, s2_x, s2_y, s2_o =  get_active_inv_direct(sess, query)
  s1_x = np.argmax(s1_x)
  s1_y = np.argmax(s1_y)
  s1_o = np.argmax(s1_o)

  s2_x = np.argmax(s2_x)
  s2_y = np.argmax(s2_y)
  #s2_o = np.argmax(s2_o)

  if s1_x == s1_x_ and s1_y == s1_y_ and s1_o == np.argmax(s1_o_)\
     and s2_x == s2_x_ and s2_y == s2_y_:
    cor_active += 1

  print "avg rand ", cor_rand / i 
  print "avg acti ", cor_active / i 
