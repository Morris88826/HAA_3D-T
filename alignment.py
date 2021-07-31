import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from libs.alignment.skeleton import Skeleton, normalize_skeleton, Skeleton3d
from libs.alignment.find_rotation import rotation_matrix_3d
import glob
import json
import tqdm
import os
import matplotlib.cm as cm
import argparse
from libs.visualize import plot_2d, plot_3d, plot_camera, plot_skeleton_3d, plot_confusion_matrix
from libs.util import load_skeletons
import time
from libs.DTW import dynamic_time_warp
from libs.images_to_vid import convert
from PIL import Image

def align(all_skeletons_3d):
    # all_skeletons_3d.shape == [num_videos, num_frames, 17, 3]
    rf_skeletons_3d = np.zeros((len(all_skeletons_3d), 17, 3))
    for i in range(len(all_skeletons_3d)):
        rf_skeletons_3d[i] = all_skeletons_3d[i][0]

    n_rf_skeletons_3d = normalize_skeleton(rf_skeletons_3d)

    n_skeletons_3d = normalize_skeleton(rf_skeletons_3d)
    limbs = Skeleton3d(n_skeletons_3d[0], is_cartesian=True).get_spherical_joints()[:, 0]

    
    reference_skeleton_3d = n_skeletons_3d[0]

    aligned_skeletons = []
    for i in range(n_skeletons_3d.shape[0]):
        rotated_input_skeletons_3d = all_skeletons_3d[i]
        
        # Calculate rotation 
        _, R  = rotation_matrix_3d(n_skeletons_3d[i], reference_skeleton_3d)
        camera_rotation_matrix = np.linalg.inv(R) # input to reference

        # Make equal length and align
        for j in range(rotated_input_skeletons_3d.shape[0]):
            spherical = Skeleton3d(rotated_input_skeletons_3d[j], is_cartesian=True).get_spherical_joints()
            spherical[:, 0] = limbs
            rotated_input_skeletons_3d[j] = Skeleton3d(spherical, is_cartesian=False).get_cartesian_joints()

            rotated_input_skeletons_3d[j] = np.matmul(np.linalg.inv(camera_rotation_matrix), rotated_input_skeletons_3d[j].T).T
        
        aligned_skeletons.append(rotated_input_skeletons_3d)
    
    return aligned_skeletons



def visualize_alignment(input_skeleton_3d, reference_skeleton_3d, camera_rotation_matrix=None):
    
    if camera_rotation_matrix is None:
        _, R  = rotation_matrix_3d(input_skeleton_3d, reference_skeleton_3d)
        camera_rotation_matrix = np.linalg.inv(R) # input to reference
    fig1 = plt.figure(figsize=(10, 10),dpi=50)
    ax = fig1.add_subplot(111, projection="3d")
    plot_3d(reference_skeleton_3d, ax)
    reference_camera_position = np.array([0,0,-1])
    selected_camera_position = np.matmul(np.linalg.inv(camera_rotation_matrix), reference_camera_position.T).T
    plot_camera(ax, reference_camera_position, np.identity(3), color='blue')
    plot_camera(ax, selected_camera_position, camera_rotation_matrix, color='red')


    fig2 = plt.figure(2, figsize=(20, 10),dpi=50)
    ax2 = fig2.add_subplot(221)
    plot_2d(reference_skeleton_3d[:, :-1], ax2)
    ax2.set(title='Reference: Default View')

    ax2 = fig2.add_subplot(222)
    rotated_reference_skeleton_3d = np.matmul(camera_rotation_matrix, reference_skeleton_3d.T).T
    plot_2d(rotated_reference_skeleton_3d[:, :-1], ax2)
    ax2.set(title='Reference: Input Camera View')

    ax2 = fig2.add_subplot(223)
    plot_2d(input_skeleton_3d[:, :-1], ax2)
    ax2.set(title='Input: Default View')

    ax2 = fig2.add_subplot(224)
    rotated_input_skeleton_3d = np.matmul(np.linalg.inv(camera_rotation_matrix), input_skeleton_3d.T).T
    plot_2d(rotated_input_skeleton_3d[:, :-1], ax2)
    ax2.set(title='Input: Aligned View')
    
    return


def save_aligned_results(aligned_skeletons, class_name):
    
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if not os.path.exists('./results/aligned'):
        os.mkdir('./results/aligned')

    if not os.path.exists('./results/aligned/{}'.format(class_name)):
        os.mkdir('./results/aligned/{}'.format(class_name))
    
    out_path = './results/aligned/{}'.format(class_name)
    
    for i in range(len(aligned_skeletons)):
        np.save(out_path+'/{}_{:03d}.npy'.format(class_name, i), aligned_skeletons[i])

    print('Done Saving.')
    return


def main(class_name):
    # Save align skeleton throughout the 20 videos
    all_skeletons_2d, all_skeletons_3d = load_skeletons(class_name)
    start_t = time.time()
    aligned_skeletons = align(all_skeletons_3d)
    print('Aligning time: {:.3f}'.format(time.time()-start_t))
    save_aligned_results(aligned_skeletons, class_name)

def plot_matching(matching, class_name, vid_1, vid_2, save=False):

    if save:
        if not os.path.exists('./results'):
            os.mkdir('./results')

        if not os.path.exists('./results/matching'):
            os.mkdir('./results/matching')

        out_path = './results/matching/{}_{}_{}'.format(class_name, vid_1, vid_2)
        if not os.path.exists(out_path):
            os.mkdir(out_path)

    image_root = './dataset/raw/{}'.format(class_name)
    for i in tqdm.tqdm(range(matching.shape[0])):
        
        fig = plt.figure(figsize = (10,7))
        ax1 = fig.add_subplot(121)
        image_1_fid = matching[i, 0]    
        image_1_path = image_root+'/{}_{:03d}/{:04d}.png'.format(class_name, vid_1, image_1_fid+1)
        ax1.imshow(np.array(Image.open(image_1_path)))
        ax1.set(title='Video {}, Frame {}'.format(vid_1, image_1_fid+1))

        ax2 = fig.add_subplot(122)
        image_2_fid = matching[i, 1]    
        image_2_path = image_root+'/{}_{:03d}/{:04d}.png'.format(class_name, vid_2, image_2_fid+1)
        ax2.imshow(np.array(Image.open(image_2_path)))
        ax2.set(title='Video {}, Frame {}'.format(vid_2, image_2_fid+1))

        if not save:
            plt.show()
        else:
            fig.savefig(out_path+'/{:04d}.png'.format(i))
            plt.close()

    if save:
        convert(out_path)
    return

def DTW_test(class_name, vid_1, vid_2):
    all_aligned_skeletons = []
    for vid,  filename in enumerate(sorted(glob.glob('./results/aligned/{}/*'.format(class_name)))):
        aligned_skeletons = np.load(filename)
        all_aligned_skeletons.append(aligned_skeletons)
    
        # for i in range(aligned_skeletons.shape[0]):
        #     fig = plot_skeleton_3d(aligned_skeletons[i], camera_idx=4, suptitle='Video {}'.format(vid))
            
        #     plt.show()
        #     break

    sequence1 = all_aligned_skeletons[vid_1]
    sequence2 = all_aligned_skeletons[vid_2]

    D, trace, cost = dynamic_time_warp(sequence1, sequence2)

    plot_matching(trace, class_name, vid_1, vid_2, save=True)

    trace_matrix = np.zeros_like(D[1:, 1:])
    trace_matrix[trace[:, 0], trace[:, 1]] = 1
    plot_confusion_matrix(trace_matrix, vid_1, vid_2)
    plt.show()
    


if __name__ == '__main__':
    class_name = 'beer_pong_throw'

    # main(class_name)
    DTW_test(class_name, 0, 8)