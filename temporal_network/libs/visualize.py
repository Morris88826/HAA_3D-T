import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from .skeleton import Skeleton, Skeleton2d, Skeleton3d
from .util import get_range_lim

def visualize_in_vid(joints, images=None, hidden={}, out_dir = './results', video_name = 'test', config=None):

    is_2d = True if joints.shape[-1] == 2 else False

    print('Generating {} ~'.format(video_name))
    if config is not None:
        config['range_lim'] = get_range_lim(joints)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    out_dir_tmp = out_dir + '/{}'.format(video_name)
    if os.path.exists(out_dir_tmp):
        os.system('rm -rf {}'.format(out_dir_tmp))
    os.mkdir(out_dir_tmp)

    for i in tqdm(range(joints.shape[0])):
        if i in hidden.keys():
            visualize(joints[i], hidden_idx=hidden[i], out_dir=out_dir_tmp, name='{:03d}'.format(i), config=config, is_2d=is_2d)
        else:
            visualize(joints[i], out_dir=out_dir_tmp, name='{:03d}'.format(i), config=config, is_2d=is_2d)

    create_video(out_dir_tmp, out_dir+'/{}.mp4'.format(video_name))

    return

def visualize_depth_map(depth_map, out_dir = './results', config=None):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    out_dir_tmp = out_dir + '/depth_map'
    if not os.path.exists(out_dir_tmp):
        os.mkdir(out_dir_tmp)

    out_dir = out_dir_tmp + '/{}_{}'.format(config['class_name'], config['video_idx'])
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for i in range(depth_map.shape[0]):
        im = plt.imshow(depth_map[i])
        plt.title('Depth Map {:04d}'.format(i))
        plt.colorbar(im)
        plt.savefig(out_dir+'/{:04d}.png'.format(i))
        plt.close()

    return

def visualize(joints, image=None, hidden_idx=None, is_2d=True, out_dir = None, name=None, config=None):
    dict_idx_2_joint = Skeleton().dict_idx_2_joint
    dict_joint_2_idx = Skeleton().dict_joint_2_idx

    color_array = [
        'darkred', 'deepskyblue', 'blue', 'cyan', 
        'olivedrab', 'lawngreen', 'lightgreen', 
        'lightcoral', 'red', 
        'deeppink', 'fuchsia', 
        'orange', 'gold', 'wheat',
        'purple', 'blueviolet', 'violet'
    ]
    
    if hidden_idx is not None:
        for hid in hidden_idx:
            color_array[hid] = 'black'

    if config is not None:
        class_name = config['class_name']
        video_idx = config['video_idx']
        range_lim = config['range_lim']
    else:
        class_name = ''
        video_idx = ''
        if is_2d:
            clip = np.amax(np.abs(joints))
            range_lim = [[0, clip], [0, clip]]
        else:
            clip = np.amax(np.abs(joints))
            range_lim = [[-clip, clip], [-clip, clip], [-clip, clip]]

    if is_2d:
        skeleton = Skeleton2d(joints)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlim(range_lim[0][0], range_lim[0][1])
        ax.set_ylim(range_lim[1][0], range_lim[1][1])

        joint = getattr(skeleton, skeleton.root_joint)
        ax.scatter(joint.cartesian_coord[0], joint.cartesian_coord[1], c=color_array[dict_joint_2_idx[joint.name]], label=joint.name)
        for body_part in skeleton.body_parts.keys():
            for i, joint_name in enumerate(skeleton.body_parts[body_part]):
                if i == 0:
                    continue
                else:
                    joint = getattr(skeleton, joint_name)
                    ax.scatter(joint.cartesian_coord[0], joint.cartesian_coord[1], c=color_array[dict_joint_2_idx[joint.name]], label=joint.name)
                    ax.plot([joint.cartesian_coord[0], joint.parent.cartesian_coord[0]], [joint.cartesian_coord[1], joint.parent.cartesian_coord[1]])
        ax.set_ylim(ax.get_ylim()[::-1])
    
    else:
        skeleton = Skeleton3d(joints)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        
        ax.set_xlim(range_lim[0][0], range_lim[0][1])
        ax.set_ylim(range_lim[1][0], range_lim[1][1])
        ax.set_zlim(range_lim[2][0], range_lim[2][1])

        joint = getattr(skeleton, skeleton.root_joint)
        ax.scatter(joint.cartesian_coord[0], joint.cartesian_coord[1], joint.cartesian_coord[2], c=color_array[dict_joint_2_idx[joint.name]], label=joint.name)
        for body_part in skeleton.body_parts.keys():
            for i, joint_name in enumerate(skeleton.body_parts[body_part]):
                if i == 0:
                    continue
                else:
                    joint = getattr(skeleton, joint_name)
                    ax.scatter(joint.cartesian_coord[0], joint.cartesian_coord[1], joint.cartesian_coord[2], c=color_array[dict_joint_2_idx[joint.name]], label=joint.name)
                    ax.plot([joint.cartesian_coord[0], joint.parent.cartesian_coord[0]], [joint.cartesian_coord[1], joint.parent.cartesian_coord[1]], [joint.cartesian_coord[2], joint.parent.cartesian_coord[2]])
    

#     ax.legend()
    if name is not None:
        ax.set_title(class_name + ' ' + str(video_idx) + ' ' + name)
    
    if out_dir is not None:
        plt.savefig('{}/{}.png'.format(out_dir, name))
    else:
        plt.show()
    
    plt.close()
    return

def create_video(folder, out_path, framerate=2):
    os.system('ffmpeg -framerate {} -pattern_type glob -i \'{}/*.png\' {}'.format(framerate, folder, out_path))