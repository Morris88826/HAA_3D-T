import glob
import os
from numpy.core.fromnumeric import sort
from libs.alignment.debug import plot_2d
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
from libs.visualize import plot_3d, plot_2d, plot_golden_circle, plot_camera
from libs.alignment.skeleton import normalize_skeleton
from libs.util import cartesian_to_spherical, rel_error, load_skeletons, get_golden_circle, find_distance
from libs.alignment.skeleton import Skeleton3d
from alignment import visualize_alignment
from libs.alignment.find_rotation import rotation_matrix_3d
import tqdm


def find_camera_rotation(reference, position):
    _, theta1, phi1 = cartesian_to_spherical(reference)
    _, theta2, phi2 = cartesian_to_spherical(position)

    d_theta = theta2-theta1
    d_phi = phi2-phi1

    Rx = np.identity(3)
    Ry = np.identity(3)
    Rz = np.identity(3)

    Ry[0,0] = math.cos(d_theta)
    Ry[0,2] = math.sin(d_theta)
    Ry[2,0] = -math.sin(d_theta)
    Ry[2,2] = math.cos(d_theta)


    Rz[:-1, :-1] = np.array(
        [[math.cos(d_phi), -math.sin(d_phi)],
        [math.sin(d_phi), math.cos(d_phi)]]
    )

    R = np.linalg.multi_dot([Rz, Rx, Ry])

    return R

def visualize_different_angle(class_name, video_idx, camera_rotation_matrix=None):
    frame_idx = 1
    _, rf_skeletons_3d = load_skeletons(class_name, frame_idx)
    n_skeletons_3d = normalize_skeleton(rf_skeletons_3d)
    r = Skeleton3d(n_skeletons_3d[video_idx]).get_spherical_joints()[:, 0]

    if camera_rotation_matrix is None:
        rf_skeleton_3d = n_skeletons_3d[0]
        input_skeleton_3d = n_skeletons_3d[video_idx]
        _, R  = rotation_matrix_3d(input_skeleton_3d, rf_skeleton_3d)
        camera_rotation_matrix = np.linalg.inv(R)
        if not os.path.exists('./results/camera_rotation'):
            os.mkdir('./results/camera_rotation')
        if not os.path.exists('./results/camera_rotation/'+class_name):
            os.mkdir('./results/camera_rotation/'+class_name)
        out_path = './results/camera_rotation/{}/{}_{:03d}.npy'.format(class_name, class_name, video_idx)
        np.save(out_path, camera_rotation_matrix)
        print('Save camera rotation matrix')
       

    normalized_all_skeletons = []
    skeleton3d_root = './dataset/joints3d/{}/{}_{:03d}'.format(class_name, class_name, video_idx) 
    for filename in sorted(glob.glob(skeleton3d_root+'/*.npy')):
        skeleton3d = np.load(filename)
        spherical_coords = Skeleton3d(skeleton3d).get_spherical_joints()
        spherical_coords[:, 0] = r
        normalized_all_skeletons.append(Skeleton3d(spherical_coords, is_cartesian=False).get_cartesian_joints())
    
    normalized_all_skeletons = np.array(normalized_all_skeletons)


    camera_position, _ = get_golden_circle()
    default_camera_position = np.array([0,0,-1])      
    for camera_idx in tqdm.tqdm(range(camera_position.shape[0])):
        for i in range(normalized_all_skeletons.shape[0]):
            rf_skeleton_3d = normalized_all_skeletons[i]

            rotated_rf_skeleton_3d = np.matmul(np.linalg.inv(camera_rotation_matrix), rf_skeleton_3d.T).T

            selected_camera_position = camera_position[camera_idx]
            
            # Plot
            fig1 = plt.figure(1, figsize=(20, 10),dpi=50)
            ax1= fig1.add_subplot(121, projection="3d")
            plot_golden_circle(ax1)
            plot_3d(rotated_rf_skeleton_3d, ax1)
            plot_camera(ax1, default_camera_position, np.identity(3), color='blue')

 
            camera_rotation = find_camera_rotation(default_camera_position, selected_camera_position) # default to selected
            assert rel_error(np.matmul(camera_rotation, default_camera_position), selected_camera_position) < 1e-6


            rotated_reference_skeleton_3d = np.matmul(np.linalg.inv(camera_rotation), rotated_rf_skeleton_3d.T).T
            plot_camera(ax1, selected_camera_position, np.linalg.inv(camera_rotation), color='red')

            ax1 = fig1.add_subplot(122)
            plot_2d(rotated_reference_skeleton_3d[:, :-1], ax1)
            ax1.set(title='View {}'.format(camera_idx))   

            if not os.path.exists('./results/views'):
                os.mkdir('./results/views')
            if not os.path.exists('./results/views/'+class_name):
                os.mkdir('./results/views/'+class_name)
            out_path = './results/views/{}/{}_{:03d}'.format(class_name, class_name, video_idx)
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            if not os.path.exists(out_path+'/view_{}'.format(camera_idx)):
                os.mkdir(out_path+'/view_{}'.format(camera_idx))
                
            fig1.savefig(out_path+'/view_{}/{:04d}.png'.format(camera_idx, i+1))
            plt.close()


def main():
    # load skeleton
    class_name = 'abseiling'
    frame_idx = 0
    rf_skeletons_2d, rf_skeletons_3d = load_skeletons(class_name, frame_idx)
    
    n_skeletons_2d = normalize_skeleton(rf_skeletons_2d)
    n_skeletons_3d = normalize_skeleton(rf_skeletons_3d)

    # check if having same limbs
    sk1 = Skeleton3d(n_skeletons_3d[0])
    sk2 = Skeleton3d(n_skeletons_3d[1])
    assert rel_error(sk1.get_spherical_joints()[:,0], sk2.get_spherical_joints()[:,0]) < 1e-5

    
    rf_skeleton_2d = n_skeletons_2d[0]
    rf_skeleton_3d = n_skeletons_3d[0]
    camera_position, _ = get_golden_circle()
    default_camera_position = np.array([0,0,-1])
    
    for i in range(5):
        selected_camera_idx = i
        # Plot
        fig1 = plt.figure(1, figsize=(20, 10),dpi=50)
        ax1= fig1.add_subplot(121, projection="3d")
        plot_golden_circle(ax1)
        plot_3d(rf_skeleton_3d, ax1)
        plot_camera(ax1, default_camera_position, np.identity(3), color='blue')

        for i in range(camera_position.shape[0]):
            selected_camera_position = camera_position[i]
            camera_rotation = find_camera_rotation(default_camera_position, selected_camera_position) # default to selected
            assert rel_error(np.matmul(camera_rotation, default_camera_position), selected_camera_position) < 1e-6

            if i == selected_camera_idx:
                rotated_reference_skeleton_3d = np.matmul(np.linalg.inv(camera_rotation), rf_skeleton_3d.T).T
                plot_camera(ax1, selected_camera_position, np.linalg.inv(camera_rotation), color='red')
            else:
                # plot_camera(ax1, selected_camera_position, np.linalg.inv(camera_rotation), color='black')
                pass

        ax1 = fig1.add_subplot(122)
        plot_2d(rotated_reference_skeleton_3d[:, :-1], ax1)
        ax1.set(title='View {}'.format(selected_camera_idx))   

        # fig2 = plt.figure(2, figsize=(10, 10),dpi=50)
        # ax2 = fig2.add_subplot(111)
        # plot_2d(rf_skeleton_3d[:, :-1], ax2)
        # ax2.set(title='Reference View'.format(selected_camera_idx)) 

        plt.show()

    video_idx = 3
    input_skeleton_2d = n_skeletons_2d[video_idx]
    input_skeleton_3d = n_skeletons_3d[video_idx]

    visualize_alignment(input_skeleton_3d, rf_skeleton_3d)
    plt.show()


if __name__ == '__main__':
    class_name = 'bench_dip'
    video_idx = 5
    camera_rotation_matrix = None
    if os.path.exists('./results/camera_rotation/{}/{}_{:03d}.npy'.format(class_name, class_name, video_idx)):
        camera_rotation_matrix =  np.load('./results/camera_rotation/{}/{}_{:03d}.npy'.format(class_name, class_name, video_idx))

    visualize_different_angle(class_name, video_idx, camera_rotation_matrix)
   