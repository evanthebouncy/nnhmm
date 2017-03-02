from data import *
from draw import *

d_idx = np.random.randint(0, 50)

x_x, obs_x, obs_y, obs_tfs, new_ob_x, new_ob_y, new_ob_tf, imgs = gen_data()

print show_dim(x_x)
print show_dim(obs_x)
print show_dim(obs_y)
print show_dim(obs_tfs)
print show_dim(new_ob_x)
print show_dim(new_ob_y)
print show_dim(new_ob_tf)


obss = zip([np.argmax(obx[d_idx]) for obx in obs_x], 
           [np.argmax(oby[d_idx]) for oby in obs_y], 
           [obtf[d_idx] for obtf in obs_tfs])

obss = [((x[0],x[1]), x[2]) for x in obss]

print "hidden number value ", np.argmax(x_x[d_idx])
draw_obs(obss, "test_obs.png")

img = imgs[d_idx]

draw_orig(img, "test_orig.png")
print img

black_white(img)

