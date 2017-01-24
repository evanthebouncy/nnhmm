from active_learning import *

# disable the random hit
RAND_HIT = 0.0

sess = load_model("model.ckpt")


hid_x = gen_X()
query = mk_query(hid_x)
rand_inv = get_random_inv(sess, query) 
acti_inv = get_active_inv(sess, query) 

# print rand_inv

draw_ship(hid_x, "orig.png")
draw_all_preds(rand_inv[20][1], "rand_inv.png")
draw_all_preds(acti_inv[20][1], "acti_inv.png")

draw_trace(acti_inv, "acti_inv_tr.png")

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
