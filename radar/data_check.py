from data import *

X = gen_X()
O, O_lab = gen_O(X)
vec_O = vectorize(O)

print X
print O
print O_lab
print vec_O

x_x, x_y, obs_x, obs_y, obs_tfs, new_ob_x, new_ob_y, new_ob_tf = gen_data()

draw_coord((x_x[0], x_y[0]), "test_X.png")
draw_coord((new_ob_x[0], new_ob_y[0]), "test_new_ob.png", new_ob_tf[0])

for xx, yy in zip(new_ob_x, new_ob_tf):
  print np.argmax(xx), yy
