import numpy as np
from .util import get_range_lim, calculate_camera_matrix, project_3d_2_2d
from .skeleton import Skeleton, Skeleton2d
from .visualize import visualize
from find_rotation_matrix import rotation_matrix
import matplotlib.pyplot as plt
import tqdm

def calculate_depth_map(image_size, points, points_3d):
    print('Calculating depth map ~')
    lim = np.array(get_range_lim(points)).astype(np.int)
    
    start_x = min(0, lim[0][0])
    start_y = min(0, lim[0][1])
    end_x = max(image_size[1], lim[0][1])
    end_y = max(image_size[0], lim[1][1])

    width = end_x - start_x
    height = end_y - start_y

    offset_x = 0 if start_x > 0 else abs(start_x)
    offset_y = 0 if start_y > 0 else abs(start_y)
    
    depth_map = np.zeros((points.shape[0], height, width))

    for i in tqdm.tqdm(range(depth_map.shape[0])):
        output = rotation_matrix(points_3d[i], points[i])
        output[:, 2] = output[:, 2]
        coord = output[:, :2].astype(np.int)
        depth_map[i, coord[:, 1] + offset_y, coord[:, 0] + offset_x] = output[:, 2]
        # visualize(output[:, :], out_dir='./', is_2d=False, name='projected')
        # visualize(points[i], out_dir='./', name='gt')

        skeleton = Skeleton2d(output[:,:2])
        for body_part in skeleton.body_parts.keys():
            for j, joint_name in enumerate(skeleton.body_parts[body_part]):
                if j == 0:
                    continue
                else:
                    joint = getattr(skeleton, joint_name)
                    line = joint.cartesian_coord - joint.parent.cartesian_coord
                    
                    base = np.argmax(np.abs(line))
                        
                    d =  output[skeleton.dict_joint_2_idx[joint.name], 2] - output[skeleton.dict_joint_2_idx[joint.parent.name], 2]                    

                    if base == 0:
                        m = line[1]/line[0]
                        if abs(m) > 1e6:
                            raise NotImplementedError
                        start_x = int(joint.parent.cartesian_coord[0])
                        end_x = int(joint.cartesian_coord[0])
                        dir = 1 if end_x - start_x > 0 else -1
                        dd = d/(end_x-start_x)
                        start_y = int(joint.parent.cartesian_coord[1])
                        
                        for idx, k in enumerate(range(start_x, end_x, dir)):
                            x = k
                            y = int(m*(k-start_x) + start_y)
                            depth_map[i, y+offset_y, x+offset_x] = output[skeleton.dict_joint_2_idx[joint.parent.name], 2] + dd*idx
                    else:
                        m = line[0]/line[1]
                        if abs(m) > 1e6:
                            raise NotImplementedError
                        start_y = int(joint.parent.cartesian_coord[1])
                        end_y = int(joint.cartesian_coord[1])
                        
                        dir = 1 if end_y - start_y > 0 else -1

                        dd = d/(end_y-start_y)
                        start_x = int(joint.parent.cartesian_coord[0])

                        for idx, k in enumerate(range(start_y, end_y, dir)):
                            y = k
                            x = int(m*(k-start_y) + start_x)
                            depth_map[i, y+offset_y, x+offset_x] = output[skeleton.dict_joint_2_idx[joint.parent.name], 2] + dd*idx

    return depth_map
