import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from .skeleton import Skeleton
import json


video_name = 'burpee_005'
frame_idx = 1


class_name = video_name.split('_')[0]
joints2d_path = './dataset/joints2d/{}/{}/{:04d}.json'.format(class_name, video_name, frame_idx)
with open(joints2d_path, 'rb') as jsonfile:
    data = json.load(jsonfile)
    joints2d = np.array(data).reshape((17,-1))

joints3d_path = joints2d_path.replace('joints2d', 'joints3d').replace('json', 'npy')
joints3d = np.load(joints3d_path)



bones_indices = [
    [[0,4],[4,5],[5,6],[8,11],[11,12],[12,13]], # left -> pink
    [[0,1],[1,2],[2,3],[8,14],[14,15],[15,16]], # right -> blue
    [[0,7],[7,8],[8,9],[9,10]] # black
] # left-> pink, right->blue

fig = plt.figure(0, figsize=(10, 10), dpi=50)
ax = fig.add_subplot(111, projection="3d")


max_range = 1000
ax.set(xlim = [-max_range, max_range], ylim=[-max_range, max_range], zlim=[max_range,-max_range])
# ax.view_init(elev=180, azim=0)

for _, bones in enumerate(bones_indices):
    for bone in (bones):
        start = bone[0]
        end = bone[1]
        x0, y0, z0 = list(joints3d[start])
        x1, y1, z1 = list(joints3d[end])
        ax.plot([x0, x1], [y0, y1], [z0, z1])
# draw dots
sk = Skeleton()
for i in range(17):
    if sk.dict_idx_2_joint[i] in (sk.body_parts['torso'] + sk.body_parts['head']):
        ax.scatter(joints3d[i, 0], joints3d[i, 1], joints3d[i, 2], c='b')
    elif sk.dict_idx_2_joint[i] in (sk.body_parts['left_leg'] + sk.body_parts['left_arm']):
        ax.scatter(joints3d[i, 0], joints3d[i, 1], joints3d[i, 2], c='r')
    elif sk.dict_idx_2_joint[i] in (sk.body_parts['right_leg'] + sk.body_parts['right_arm']):
        ax.scatter(joints3d[i, 0], joints3d[i, 1], joints3d[i, 2], c='g')
    else:
        raise NotImplementedError
        

fig, ax = plt.subplots()

if max_range is None:
    max_range = np.amax(np.abs(joints2d))

ax.set(xlim = [0, 1024], ylim=[720, 0])

for _, bones in enumerate(bones_indices):
    for bone in (bones):
        start = bone[0]
        end = bone[1]

        x0, y0, = list(joints2d[start])
        x1, y1, = list(joints2d[end])
        ax.plot([x0, x1], [y0, y1])

# draw dots
for i in range(17):
    if sk.dict_idx_2_joint[i] in (sk.body_parts['torso'] + sk.body_parts['head']):
        ax.scatter(joints2d[i, 0], joints2d[i, 1], c='b')
    elif sk.dict_idx_2_joint[i] in (sk.body_parts['left_leg'] + sk.body_parts['left_arm']):
        ax.scatter(joints2d[i, 0], joints2d[i, 1], c='r')
    elif sk.dict_idx_2_joint[i] in (sk.body_parts['right_leg'] + sk.body_parts['right_arm']):
        ax.scatter(joints2d[i, 0], joints2d[i, 1], c='g')
    else:
        raise NotImplementedError
        

plt.show()