from data import *

X = gen_X()
print X
draw_ship(X, "test_X.png")
Obs = [gen_O(X) for _ in range(20)]
print Obs

draw_obs(Obs, "test_Os.png")

s1_x, s1_y, s1_o, s2_x, s2_y, s2_o, obs_x, obs_y, obs_tfs, new_ob_x, new_ob_y, new_ob_tf = gen_data()

print show_dim(s1_x)
print show_dim(s1_y)
print show_dim(s1_o)
print show_dim(s2_x)
print show_dim(s2_y)
print show_dim(s2_o)
print show_dim(obs_x)
print show_dim(obs_y)
print show_dim(obs_tfs)
print show_dim(new_ob_x)
print show_dim(new_ob_y)
print show_dim(new_ob_tf)
# 
# draw_coord((new_ob_x[0], new_ob_y[0]), "test_new_ob.png", new_ob_tf[0])
# 
# draw_annotate([1,2,3], [1,2,3], [True,5,6], "haha.png")

