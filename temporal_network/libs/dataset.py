import numpy as np
import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader

class TemporalDataV2(Dataset):
    def __init__(self, type=None, data_path='./dataset/temporal_data.pkl'):
        with open(data_path, 'rb') as datafile:
            datafile = pkl.load(datafile)

        self.type = type
        self.data = datafile[type]
        self.length = len(self.data)
        self.time_frames = 9
        self.all_frames = 0
        for d in (self.data):
            self.all_frames += d['data']['2d_occ'].shape[0]
        self.trial_times = self.all_frames // (self.length)

    def __getitem__(self, idx):

        data = self.data[idx]
        class_name = data['class']
        video_idx = data['vid']
        joints2d_occ = data['data']['2d_occ']
        joints2d_gt = data['data']['2d_gt']
        joints3d_gt = data['data']['3d']

        num_frames = joints2d_occ.shape[0]
        selected_idx = np.random.randint(num_frames)        
        joints2d_occ_tf = np.repeat(joints2d_occ[selected_idx][np.newaxis, :], self.time_frames, axis=0)
        joints2d_gt = joints2d_gt[selected_idx]
        joints3d_gt = joints3d_gt[selected_idx]

        for idx, i in enumerate(np.arange(self.time_frames)-(self.time_frames//2)):
            current_idx = selected_idx + i
            if current_idx >= 0 and current_idx < num_frames:
                joints2d_occ_tf[idx] = joints2d_occ[current_idx]

        normalized_joints2d_occ = self.normalization(joints2d_occ_tf)
        normalized_joints2d_gt = self.normalization(joints2d_gt)
        normalized_joints3d_gt = self.normalization(joints3d_gt)
        
        return normalized_joints2d_occ, normalized_joints2d_gt, normalized_joints3d_gt
        

    def __len__(self):
        return self.length
    

    def normalization(self, joints):

        if joints.ndim == 3:
            displacement = np.copy(joints[self.time_frames//2, 0, :2])
            joints[:, :, :2] -= displacement
            scaler = np.max(np.abs(joints[:, :, :2]))
            joints[:, :, :2] /= scaler
            

        else:
            displacement = np.copy(joints[0])
            joints -= displacement
            scaler = np.max(np.abs(joints))
            joints /= scaler

        return joints


class TemporalData():
    def __init__(self, data_path='./dataset/temporal_data.pkl'):
        with open(data_path, 'rb') as datafile:
            datafile = pkl.load(datafile)

        self.train_data = datafile['train']
        self.val_data = datafile['val']
        self.test_data = datafile['test']
        self.num_train = len(self.train_data)
        self.num_val = len(self.val_data)
        self.num_test = len(self.test_data)

        self.load_type = 'train'

    def load_data(self, index):
        data = getattr(self, self.load_type+'_data')[index]
        
        class_name = data['class']
        video_idx = data['vid']
        return class_name, video_idx, data['data']
    
    def dataloader(self, index, batch_size=10):
        data = getattr(self, self.load_type+'_data')[index]
        data_package = data['data']
        dataset = _TemporalData(data_package)
        return DataLoader(dataset, batch_size=batch_size)

    def get_length(self):
        return len(getattr(self, self.load_type+'_data'))

class _TemporalData(Dataset):
    def __init__(self, data_package, time_frames=9):
        super(_TemporalData, self).__init__()
        self.joints2d_occ = data_package['2d_occ']
        self.joints2d_gt = data_package['2d_gt']
        self.joints3d_gt = data_package['3d']
        self.time_frames = time_frames
        self.length = self.joints2d_occ.shape[0]

    def __getitem__(self, idx):
        joints2d_ori = np.copy(self.joints2d_occ[idx])
        joints2d_occ = np.expand_dims(self.joints2d_occ[idx], axis=0)
        joints2d_gt = self.joints2d_gt[idx]
        joints3d_gt = self.joints3d_gt[idx]

        joints2d_occ = np.repeat(joints2d_occ, self.time_frames, axis=0)
        # joints3d_gt = np.rep/eat(joints3d_gt, self.time_frames, axis=0)
        
        for i, j in enumerate(range(-self.time_frames//2+1, self.time_frames//2 + 1)):
            if idx + j >= 0 and idx + j < self.length:
                joints2d_occ[i] = self.joints2d_occ[idx+j]
                # joints3d_gt[i] = self.joints3d_gt[idx+j]

        joints2d_occ = self.normalization(joints2d_occ)
        joints2d_gt = self.normalization(joints2d_gt)
        joints3d_gt = self.normalization(joints3d_gt)

        return joints2d_occ, joints2d_gt, joints3d_gt, joints2d_ori

    def normalization(self, joints):

        if joints.ndim == 3:
            displacement = np.copy(joints[self.time_frames//2, 0, :2])
            joints[:, :, :2] -= displacement
            scaler = np.max(np.abs(joints[:, :, :2]))
            joints[:, :, :2] /= scaler
            

        else:
            displacement = np.copy(joints[0])
            joints -= displacement
            scaler = np.max(np.abs(joints))
            joints /= scaler

        return joints

    def __len__(self):
        return self.joints2d_occ.shape[0]

if __name__ == '__main__':
    train_data = TemporalDataV2(type='train')
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

    joints2d_occ, joints2d_gt, joints3d_gt = next(iter(train_loader))

    print(joints2d_occ.shape)
    print(joints2d_gt.shape)
    print(joints3d_gt.shape)