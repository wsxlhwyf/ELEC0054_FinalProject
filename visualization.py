import os
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"
from matplotlib import pyplot as plt
from dataloader import load_data
import random


"""Select k visualizations from the train"""
all_data, all_label = load_data('train')
k = 10 
samples_index = random.sample(range(9840), k)
samples = all_data[samples_index]

for i in range(k):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(samples[i, :, 0],  samples[i, :, 1], samples[i, :, 2], c=samples[i, :, 2], cmap='Spectral')
    plt.savefig('./image/train_' + str(i) + '.png')



"""Select k visualizations from the test"""
# all_data, all_label = load_data('test')
# k = 10 # 随机挑十张可视化
# samples_index = random.sample(range(2468), k)
# samples = all_data[samples_index]
#
# for i in range(k):
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(samples[i, :, 0],  samples[i, :, 1], samples[i, :, 2], c=samples[i, :, 2], cmap='Spectral')
#     plt.savefig('./image/test_' + str(i) + '.png')