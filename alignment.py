import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from libs.alignment.skeleton import Skeleton, normalize_skeleton
from libs.alignment.find_rotation import rotation_matrix_3d
import glob
import json
import tqdm
import os
import matplotlib.cm as cm
import argparse

def visualize_camera(reference_joints3d, camera_rotation_matrices):
    bones_indices = [
        [[0,4],[4,5],[5,6],[8,11],[11,12],[12,13]], # left -> pink
        [[0,1],[1,2],[2,3],[8,14],[14,15],[15,16]], # right -> blue
        [[0,7],[7,8],[8,9],[9,10]] # black
    ] # left-> pink, right->blue

    fig = plt.figure(0, figsize=(10, 10), dpi=50)
    ax = fig.add_subplot(111, projection="3d")


    max_range = 1
    ax.set(xlim = [-max_range, max_range], ylim=[-max_range, max_range], zlim=[max_range,-max_range])
    # ax.view_init(elev=180, azim=0)

    for _, bones in enumerate(bones_indices):
        for bone in (bones):
            start = bone[0]
            end = bone[1]

            x0, y0, z0 = list(reference_joints3d[start])
            x1, y1, z1 = list(reference_joints3d[end])
            ax.plot([x0, x1], [y0, y1], [z0, z1])
    # draw dots
    sk = Skeleton()
    for i in range(17):
        if sk.dict_idx_2_joint[i] in (sk.body_parts['torso'] + sk.body_parts['head']):
            ax.scatter(reference_joints3d[i, 0], reference_joints3d[i, 1], reference_joints3d[i, 2], c='b')
        elif sk.dict_idx_2_joint[i] in (sk.body_parts['left_leg'] + sk.body_parts['left_arm']):
            ax.scatter(reference_joints3d[i, 0], reference_joints3d[i, 1], reference_joints3d[i, 2], c='r')
        elif sk.dict_idx_2_joint[i] in (sk.body_parts['right_leg'] + sk.body_parts['right_arm']):
            ax.scatter(reference_joints3d[i, 0], reference_joints3d[i, 1], reference_joints3d[i, 2], c='g')
        else:
            raise NotImplementedError

    # draw camera
    default_camera = np.array([0, -1, 0])
    default_camera_x = np.array([0.1, 0, 0])
    default_camera_y =np.array([0, 0.1, 0])
    default_camera_z = np.array([0, 0, 0.1])
    ax.scatter(default_camera[0], default_camera[1], default_camera[2], c='black')
    ax.plot([default_camera[0], (default_camera+default_camera_x)[0]], [default_camera[1], (default_camera+default_camera_x)[1]], [default_camera[2], (default_camera+default_camera_x)[2]], c='r')
    ax.plot([default_camera[0], (default_camera+default_camera_y)[0]], [default_camera[1], (default_camera+default_camera_y)[1]], [default_camera[2], (default_camera+default_camera_y)[2]], c='g')
    ax.plot([default_camera[0], (default_camera+default_camera_z)[0]], [default_camera[1], (default_camera+default_camera_z)[1]], [default_camera[2], (default_camera+default_camera_z)[2]], c='b')


    colors = cm.rainbow(np.linspace(0, 1, len(camera_rotation_matrices)))
    for idx, i in enumerate(camera_rotation_matrices.keys()):

        camera_points = []
        for j in range(camera_rotation_matrices[i].shape[0]):
            camera_rotation = camera_rotation_matrices[i][j]
            camera_point = np.matmul(camera_rotation, default_camera.T).T
            camera_points.append(camera_point)
            camera_x = np.matmul(camera_rotation, default_camera_x.T).T
            camera_y = np.matmul(camera_rotation, default_camera_y.T).T
            camera_z = np.matmul(camera_rotation, default_camera_z.T).T

            ax.plot([camera_point[0], (camera_point+camera_x)[0]], [camera_point[1], (camera_point+camera_x)[1]], [camera_point[2], (camera_point+camera_x)[2]], c='r')
            ax.plot([camera_point[0], (camera_point+camera_y)[0]], [camera_point[1], (camera_point+camera_y)[1]], [camera_point[2], (camera_point+camera_y)[2]], c='g')
            ax.plot([camera_point[0], (camera_point+camera_z)[0]], [camera_point[1], (camera_point+camera_z)[1]], [camera_point[2], (camera_point+camera_z)[2]], c='b')
        
        camera_points = np.array(camera_points)
        ax.scatter(camera_points[:, 0], camera_points[:, 1], camera_points[:, 2], c=colors[idx].reshape(1,-1), label='frame {}'.format(i))
    # plot sphere
    radius = 1
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
    x = radius*np.cos(u)*np.sin(v)
    y = radius*np.sin(u)*np.sin(v)
    z = radius*np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray")

    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.legend()
    plt.show()


def visualize(input_joints3d, input_joints2d, reference_joints3d, reference_joints2d, camera_rotation_matrices, video_idx=1):

    input_joints3d = input_joints3d[video_idx]
    input_joints2d = input_joints2d[video_idx]

    bones_indices = [
        [[0,4],[4,5],[5,6],[8,11],[11,12],[12,13]], # left -> pink
        [[0,1],[1,2],[2,3],[8,14],[14,15],[15,16]], # right -> blue
        [[0,7],[7,8],[8,9],[9,10]] # black
    ] # left-> pink, right->blue

    fig = plt.figure(0, figsize=(10, 10), dpi=50)
    ax = fig.add_subplot(111, projection="3d")


    max_range = 1
    ax.set(xlim = [-max_range, max_range], ylim=[-max_range, max_range], zlim=[max_range,-max_range])
    # ax.view_init(elev=180, azim=0)

    # reference_joints3d = input_joints3d
    for _, bones in enumerate(bones_indices):
        for bone in (bones):
            start = bone[0]
            end = bone[1]

            x0, y0, z0 = list(reference_joints3d[start])
            x1, y1, z1 = list(reference_joints3d[end])
            ax.plot([x0, x1], [y0, y1], [z0, z1])
    
    # draw dots
    sk = Skeleton()
    for i in range(17):
        if sk.dict_idx_2_joint[i] in (sk.body_parts['torso'] + sk.body_parts['head']):
            ax.scatter(reference_joints3d[i, 0], reference_joints3d[i, 1], reference_joints3d[i, 2], c='b')
        elif sk.dict_idx_2_joint[i] in (sk.body_parts['left_leg'] + sk.body_parts['left_arm']):
            ax.scatter(reference_joints3d[i, 0], reference_joints3d[i, 1], reference_joints3d[i, 2], c='r')
        elif sk.dict_idx_2_joint[i] in (sk.body_parts['right_leg'] + sk.body_parts['right_arm']):
            ax.scatter(reference_joints3d[i, 0], reference_joints3d[i, 1], reference_joints3d[i, 2], c='g')
        else:
            raise NotImplementedError

    # draw camera
    default_camera = np.array([0, -1, 0])
    default_camera_x = np.array([0.1, 0, 0])
    default_camera_y =np.array([0, 0.1, 0])
    default_camera_z = np.array([0, 0, 0.1])
    ax.scatter(default_camera[0], default_camera[1], default_camera[2], c='black')
    ax.plot([default_camera[0], (default_camera+default_camera_x)[0]], [default_camera[1], (default_camera+default_camera_x)[1]], [default_camera[2], (default_camera+default_camera_x)[2]], c='r')
    ax.plot([default_camera[0], (default_camera+default_camera_y)[0]], [default_camera[1], (default_camera+default_camera_y)[1]], [default_camera[2], (default_camera+default_camera_y)[2]], c='g')
    ax.plot([default_camera[0], (default_camera+default_camera_z)[0]], [default_camera[1], (default_camera+default_camera_z)[1]], [default_camera[2], (default_camera+default_camera_z)[2]], c='b')


    for i in range(camera_rotation_matrices.shape[0]):

        if i != video_idx:
            continue
        camera_rotation = np.linalg.inv(camera_rotation_matrices[i])

        camera_point = np.matmul(camera_rotation, default_camera.T).T
        camera_x = np.matmul(camera_rotation, default_camera_x.T).T
        camera_y = np.matmul(camera_rotation, default_camera_y.T).T
        camera_z = np.matmul(camera_rotation, default_camera_z.T).T

        if i == video_idx:
            ax.scatter(camera_point[0], camera_point[1], camera_point[2], c='purple')
        else:
            ax.scatter(camera_point[0], camera_point[1], camera_point[2], c='orange')
        ax.plot([camera_point[0], (camera_point+camera_x)[0]], [camera_point[1], (camera_point+camera_x)[1]], [camera_point[2], (camera_point+camera_x)[2]], c='r')
        ax.plot([camera_point[0], (camera_point+camera_y)[0]], [camera_point[1], (camera_point+camera_y)[1]], [camera_point[2], (camera_point+camera_y)[2]], c='g')
        ax.plot([camera_point[0], (camera_point+camera_z)[0]], [camera_point[1], (camera_point+camera_z)[1]], [camera_point[2], (camera_point+camera_z)[2]], c='b')
    
    # plot sphere
    radius = 1
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
    x = radius*np.cos(u)*np.sin(v)
    y = radius*np.sin(u)*np.sin(v)
    z = radius*np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray")

    ax.set(xlabel='x', ylabel='y', zlabel='z')


    # Camera 2D view, First Fig
    camera_view = plt.figure(1, figsize=(20, 10), dpi=50)
    ax = camera_view.add_subplot(221)
    for _, bones in enumerate(bones_indices):
        for bone in (bones):
            start = bone[0]
            end = bone[1]

            x0, y0 = list(reference_joints2d[start])
            x1, y1 = list(reference_joints2d[end])
            ax.plot([x0, x1], [y0, y1])
    ax.scatter(reference_joints2d[:, 0], reference_joints2d[:, 1], s=1)
    ax.set_xlim(-1280,1280)
    ax.set_ylim(1280,-1280)
    ax.set_title('Reference, Default Angle')



    # Second Fig
    ax = camera_view.add_subplot(222)
    camera_rotation = camera_rotation_matrices[video_idx]
    reference_projected_joints2d = np.copy(reference_sample_3d)
    reference_projected_joints2d = (np.matmul(camera_rotation,reference_projected_joints2d.T)).T
    for _, bones in enumerate(bones_indices):
        for bone in (bones):
            start = bone[0]
            end = bone[1]

            x0, y0 = list(reference_projected_joints2d[start][[0,2]])
            x1, y1 = list(reference_projected_joints2d[end][[0,2]])
            ax.plot([x0, x1], [y0, y1])
    ax.scatter(reference_projected_joints2d[:, 0], reference_projected_joints2d[:, 2], s=1)
    ax.set_xlim(-1,1)
    ax.set_ylim(1,-1)
    ax.set_title('Reference, Video {} Angle'.format(video_idx))
    
    # Thrid Fig
    ax = camera_view.add_subplot(223)
    for _, bones in enumerate(bones_indices):
        for bone in (bones):
            start = bone[0]
            end = bone[1]

            x0, y0 = list(input_joints2d[start])
            x1, y1 = list(input_joints2d[end])
            ax.plot([x0, x1], [y0, y1])
    ax.scatter(input_joints2d[:, 0], input_joints2d[:, 1], s=1)
    ax.set_xlim(-1280,1280)
    ax.set_ylim(1280,-1280)
    ax.set_title('Video {}, Original Angle'.format(video_idx))


    # Fourth Fig
    ax = camera_view.add_subplot(224)
    camera_rotation = camera_rotation_matrices[video_idx]
    original_projected_joints2d = np.copy(input_joints3d)
    new_projected_joints2d = (np.matmul(np.linalg.inv(camera_rotation),original_projected_joints2d.T)).T
    for _, bones in enumerate(bones_indices):
        for bone in (bones):
            start = bone[0]
            end = bone[1]

            x0, y0 = list(new_projected_joints2d[start][[0,2]])
            x1, y1 = list(new_projected_joints2d[end][[0,2]])
            ax.plot([x0, x1], [y0, y1])
    ax.scatter(new_projected_joints2d[:, 0], new_projected_joints2d[:, 2], s=1)
    ax.set_xlim(-1,1)
    ax.set_ylim(1,-1)
    ax.set_title('Aligned View')

    camera_view.suptitle('Video {}'.format(video_idx))
    
    plt.show()
        

def get_camera_rotation_matrices(equalized_all_joints3d, reference_sample_3d):
    camera_rotation_matrices =  []
    for i in tqdm.tqdm(range(equalized_all_joints3d.shape[0])):
        input_sample = equalized_all_joints3d[i]
        _, R = rotation_matrix_3d(input_sample, reference_sample_3d)
        camera_rotation = np.linalg.inv(R)
        camera_rotation_matrices.append(camera_rotation)

    camera_rotation_matrices = np.array(camera_rotation_matrices)

    return camera_rotation_matrices


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', '-p', help='Name of the class')
    parser.add_argument('--save', '-s', type=bool, default=False, help='Save rotation matrix or not')
    parser.add_argument('--target_vid', '-t', type=int, default=1)
    
    args = parser.parse_args()

    
    frame_indices = [1, 5, 10]
    class_name = args.pose
    save_rotation_matrix = args.save
    target_video_idx = args.target_vid

    target_3d_folder = './dataset/joints3d/{}'.format(class_name)

    if not os.path.exists('./results'):
        os.mkdir('./results')
    if not os.path.exists('./results/camera_rotation_matrices'):
        os.mkdir('./results/camera_rotation_matrices')
    if not os.path.exists('./results/camera_rotation_matrices/{}'.format(class_name)):
        os.mkdir('./results/camera_rotation_matrices/{}'.format(class_name))


    all_joints3d = {}
    all_joints2d = {}
    equalized_all_joints2d = {}
    equalized_all_joints3d = {}

    for frame_idx in frame_indices:
        all_joints2d[frame_idx] = []
        all_joints3d[frame_idx]= []
        equalized_all_joints2d[frame_idx] = []
        equalized_all_joints3d[frame_idx]= []


    for subfolder in sorted(glob.glob(target_3d_folder+'/*')):
        for frame_idx in frame_indices:
            filename =  '{:04}.npy'.format(frame_idx)
            joints3d = np.load(subfolder+'/'+filename)
            all_joints3d[frame_idx].append(joints3d)
        
    
    target_2d_folder = target_3d_folder.replace('joints3d', 'joints2d')


    for subfolder in sorted(glob.glob(target_2d_folder+'/*')):
        
        for frame_idx in frame_indices:
            filename =  '{:04}.json'.format(frame_idx)
            with open(subfolder+'/'+filename, 'rb') as jsonfile:
                data = json.load(jsonfile)
        
            joints2d = np.array(data).reshape((17, -1))
            all_joints2d[frame_idx].append(joints2d)
    

    camera_rotation_matrices = {}
    for frame_idx in frame_indices:
        all_joints2d[frame_idx] = np.array(all_joints2d[frame_idx])
        all_joints3d[frame_idx] = np.array(all_joints3d[frame_idx])

        equalized_all_joints2d[frame_idx] = normalize_skeleton(all_joints2d[frame_idx])
        equalized_all_joints3d[frame_idx] = normalize_skeleton(all_joints3d[frame_idx])

        reference_sample_2d = equalized_all_joints2d[frame_idx][0]
        reference_sample_3d = equalized_all_joints3d[frame_idx][0]


        if save_rotation_matrix:

            #  camera_rotation_matrices: Rotate default (0,-1,0) to the aligned degree
            camera_rotation_matrices[frame_idx] = get_camera_rotation_matrices(equalized_all_joints3d[frame_idx], reference_sample_3d)
            np.save('./results/camera_rotation_matrices/{}/{:04d}.npy'.format(class_name, frame_idx), camera_rotation_matrices[frame_idx])

    if not save_rotation_matrix:
        for frame_idx in frame_indices:
            camera_rotation_matrices[frame_idx] = np.load('./results/camera_rotation_matrices/{}/{:04d}.npy'.format(class_name, frame_idx))

            reference_sample_2d = equalized_all_joints2d[frame_idx][0]
            reference_sample_3d = equalized_all_joints3d[frame_idx][0]
            visualize(equalized_all_joints3d[frame_idx], equalized_all_joints2d[frame_idx], reference_sample_3d, reference_sample_2d, camera_rotation_matrices[frame_idx], video_idx=target_video_idx)

        visualize_camera(equalized_all_joints3d[frame_indices[0]][0], camera_rotation_matrices)
