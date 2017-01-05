from active_learning import *

sess = load_model("model.ckpt")

hid_x = gen_X()
query = mk_query(hid_x)
print hid_x
print get_random_inv(sess, query) 
