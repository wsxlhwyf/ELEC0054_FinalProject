import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

def load_data(partition):
    """
    Load dataset from the .h5 file
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the current file path
    all_data =[]
    all_label=[]
    # Place the data folder in the current file path, concatenate the addresses, and find the train and test data by name
    for h5_name in glob.glob(os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        # Read data
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        # Combine data
        all_data.append(data)
        all_label.append(label)
    # Combine data again outside the loop
    all_data = np.concatenate(all_data, axis=0)
    all_label= np.concatenate(all_label, axis=0)
    return all_data, all_label

def random_point_dropout(pc, max_dropout_ratio=0.0):
    """
    The function is to perform dropout on data
    """
    # Dropout rate is difficult to control and can easily affect accuracy. It is temporarily abandoned
    dropout_ratio = np.random.random() * max_dropout_ratio # Drop rate, randomly generated, range 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0] # Filter points based on drop rate
    if len(drop_idx)> 0:
        pc[drop_idx, :]= pc[0, :] # Randomly set a portion of points in the point cloud as the position of the first point
    return pc

def translate_pointcloud(pointcloud):
    """
    The purpose of this function is to enhance data and perform operations such as rotation on point clouds
    """
    # Later, it was found that the accuracy decreased after enhancement.Temporarily abandoned.
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translate_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translate_pointcloud

class ModelNet40(Dataset):
    """
    Packaging data into torch's Dataset class
    """
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition) 
        self.num_points = num_points # The number of sampled points
        self.partition = partition # Distinguish between variables of train and test

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]  # Directly Sampled
        label = self.label[item]
        # The following is the part of data augmentation, temporarily abandoned, as it is difficult to control the impact of parameters on accuracy
        # if self.partition == 'train':
        #     # pointcloud = random_point_dropout(pointcloud) # dropout train, not test
        #     pointcloud = translate_pointcloud(pointcloud) # enhance train
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    train = ModelNet40(2048)
    test = ModelNet40(2048, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
        print(label)
        break