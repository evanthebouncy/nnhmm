from data import *

obs_x, obs_y, obs_tf, new_x, new_y, new_tf, orig_x = gen_data()

print show_dim(obs_x)
print show_dim(obs_y)
print show_dim(obs_tf)
print show_dim(new_x)
print show_dim(new_y)
print show_dim(new_tf)

print obs_x[0][0]
print obs_y[0][0]
print obs_tf[0][0]
print new_x[0]
print new_y[0]
print new_tf[0]
print orig_x[0]

