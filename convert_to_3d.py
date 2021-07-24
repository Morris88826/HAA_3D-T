import numpy as np
import torch
from temporal_network.EvoSkeleton.load_model import EvoNet 
import glob
import os
import json
from matplotlib.figure import Figure
import tqdm
import argparse


def visualize(joints3d):

    plot_3d = Figure(figsize=(4, 4), dpi=50)

    ax = plot_3d.add_subplot(111, projection="3d")
    max_range = np.amax(np.abs(joints3d))
    ax.set(xlim = [-max_range, max_range], ylim=[-max_range, max_range], zlim=[-max_range, max_range])
    # ax.view_init(elev=180, azim=0)

    bones_indices = [
        [[0,4],[4,5],[5,6],[8,11],[11,12],[12,13]], # left -> pink
        [[0,1],[1,2],[2,3],[8,14],[14,15],[15,16]], # right -> blue
        [[0,7],[7,8],[8,9],[9,10]] # black
    ] # left-> pink, right->blue

    


    for i, bones in enumerate(bones_indices):
        for bone in (bones):
            start = bone[0]
            end = bone[1]

            x0, y0, z0 = list(joints3d[start])
            x1, y1, z1 = list(joints3d[end])
            ax.plot([x0, x1], [y0, y1], [z0, z1])
    # draw dots
    ax.scatter(joints3d[:, 0], joints3d[:, 1], joints3d[:, 2])
    plot_3d.savefig('./example.png')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', '-p', help='Name of the class')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evoNet_root = './temporal_network/EvoSkeleton/examples'
    evoNet = EvoNet(root = evoNet_root).to(device)
    evoNet.eval()


    class_name = args.pose
    folder_root = './dataset/joints2d/{}'.format(class_name)

    target_root = folder_root.replace('joints2d', 'joints3d')

    if not os.path.exists('./dataset/joints3d'):
        os.mkdir('./dataset/joints3d')

    if not os.path.exists(target_root):
        os.mkdir(target_root)

    for subfolder in tqdm.tqdm(sorted(glob.glob(folder_root+'/*'))):

        # load joints2d
        target_folder = subfolder.replace('joints2d','joints3d')
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)

        for frame in sorted(glob.glob(subfolder+'/*')):
            with open(frame, 'rb') as jsonfile:
                data = np.array(json.load(jsonfile)).reshape((17,-1))

                input = torch.Tensor(data[np.newaxis][:, :, :2]).to(device)
                
                input = evoNet.normalize(input)
                output = evoNet.forward(input)
                joints3d = evoNet.afterprocessing(output)[0]

                np.save(target_folder+'/{}.npy'.format(frame.split('/')[-1].split('.')[0]), joints3d)
        
