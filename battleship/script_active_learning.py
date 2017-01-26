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

# 
# print "original hidden ", hid_x
# print "random sample ", rand_inv
# print "active learning ", acti_inv
# draw_trace(rand_inv, "rand_inv.png")
# draw_trace(acti_inv, "acti_inv.png")
# 
# assert 0
# 
# 
# dist_rand = 0
# dist_active = 0
# for i in range(1,1000):
#   hid_x = gen_X()
#   query = mk_query(hid_x)
#   rand_inv = get_random_inv(sess, query) 
#   acti_inv = get_active_inv(sess, query) 
#   dist_rand += dist(rand_inv, hid_x)
#   dist_active += dist(acti_inv, hid_x)
# 
#   print "avg dist rand ", dist_rand / i 
#   print "avg dist acti ", dist_active / i 
