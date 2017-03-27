from data import *

X = gen_X()
query = mk_query(X)

print X
print query((1,2))
print gen_O(X)

x_x, obs1, obs2, obs_tfs, new_ob_1, new_ob_2, new_ob_tf = gen_data()
