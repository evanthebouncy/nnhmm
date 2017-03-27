from active_learning import *

# disable the random hit
RAND_HIT = 0.0

sess = load_model("model.ckpt")


hid_x = gen_X()
print hid_x
query = mk_query(hid_x)
rand_inv = get_random_inv(sess, query) 
acti_inv = get_active_inv(sess, query) 

print "random sample"
s1_x, s1_y, s1_o, s2_x, s2_y, s2_o =  get_random_inv_direct(sess, query)
print s1_x, np.argmax(s1_x)
print s1_y, np.argmax(s1_y)
print s1_o
print s2_x, np.argmax(s2_x)
print s2_y, np.argmax(s2_y)
print s2_o

print "active learning"
s1_x, s1_y, s1_o, s2_x, s2_y, s2_o =  get_active_inv_direct(sess, query)
print s1_x, np.argmax(s1_x)
print s1_y, np.argmax(s1_y)
print s1_o
print s2_x, np.argmax(s2_x)
print s2_y, np.argmax(s2_y)
print s2_o



draw_ship(hid_x, "drawings/orig.png")
for i in range(OBS_SIZE+1):
  draw_all_preds(rand_inv[i][1], "drawings/rand_inv{0}.png".format(i))
  draw_all_preds(acti_inv[i][1], "drawings/acti_inv{0}.png".format(i))

draw_trace(rand_inv, "drawings/rand_inv_tr.png")
draw_trace(acti_inv, "drawings/acti_inv_tr.png")

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
