from data import *

X = gen_X()
O, O_lab = gen_O(X)
vec_O = vectorize(O)

print X
print O
print O_lab
print vec_O

x_x, x_y, obs_x, obs_y, obs_tfs, new_obs, new_ob_tfs = gen_data()

print "dim ob ", show_dim(new_obs)

draw_coord((x_x[0], x_y[0]), "test_X.png")

