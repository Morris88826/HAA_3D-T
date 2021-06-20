import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from .skeleton import Skeleton, Skeleton2d, Skeleton3d
from .util import get_range_lim

def visualize_2d_full(all_joints2d, images=None, hidden=None, out_dir=None, video_name='tmp', out_name='tmp'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    if not os.path.exists(out_dir+'/'+video_name):
        os.mkdir(out_dir+'/'+video_name)  

    if not os.path.exists(out_dir+'/'+video_name+'/'+out_name):
        os.mkdir(out_dir+'/'+video_name+'/'+out_name)  
    
    out_path = out_dir+'/'+video_name+'/'+out_name
    range_lim = get_range_lim(all_joints2d)
    for frame_id in range(all_joints2d.shape[0]):
        config = {
            'range_lim':range_lim,
            'frame_id':frame_id,
            'video_name':video_name,
            'out_path':out_path
        }

        image = images[frame_id] if images is not None else None
        hidden_idx = hidden[frame_id] if hidden is not None else None

        visualize_2d(all_joints2d[frame_id], config, image=image, hidden_idx=hidden_idx)


def visualize_2d(joints2d, config, image=None, hidden_idx=None, image_name=None, save_path=None):


    range_lim = config['range_lim']
    frame_id = config['frame_id']
    video_name = config['video_name']
    out_path = config['out_path']

    if image_name is None:
        image_name = video_name + ' {}'.format(frame_id)
    if save_path is None:
        save_path = out_path + '/{:03d}.png'.format(frame_id) 

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

    skeleton = Skeleton2d(joints2d)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if image is not None:
        range_lim[0][0] = min(0, range_lim[0][0])
        range_lim[0][1] = max(image.shape[1], range_lim[0][1])
        range_lim[1][0] = min(0, range_lim[1][0])
        range_lim[1][1] = max(image.shape[0], range_lim[1][1])
        ax.imshow(image)

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
    ax.set_title(image_name)

    fig.savefig(save_path)
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