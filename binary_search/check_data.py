from data import *

X = gen_X()
O, O_lab = gen_O(X)
# vec_O = vectorize(O)

print X
print O
print O_lab
#print vec_O

x_x, obs_x, obs_tfs, new_ob_x, new_ob_tf = gen_data()
print show_dim(x_x)
print show_dim(obs_x)
print show_dim(obs_tfs)
print show_dim(new_ob_x)
print show_dim(new_ob_tf)

#draw_coord((x_x[0], x_y[0]), "test_X.png")
#draw_coord((new_ob_x[0], new_ob_y[0]), "test_new_ob.png", new_ob_tf[0])
#
#draw_annotate([1,2,3], [1,2,3], [True,5,6], "haha.png")

