
# The methodology used in this project
In this project, a new model CNN_CombinePD is proposed based on the CNN model with point clouds input to solve the 3D object recognition problem  
In the model, four 1D CNN model are used and max-pooling, min-pooling, mean-pooling and std-pooling would apply to the CNN result to realize the classification  

# The dataset for both tasks
The datasets for this project is ModelNet40 which is downloaded from the website: https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip  
After unzip the file, all the data files need to store in the file "modelnet40_ply_hdf5_2048"  

# The environment of both tasks
Pytroch is used for this project and the downloading website is : https://pytorch.org/  
Moreover, several libraries need to be import: os, glob, h5py, numpy, matplotlib, sklearn.metrics, torch and pandas  

# Result
The results will be stored in different files:  
"confusion_matrix.csv" stores the best confusion matrix result.  
"test_acc.txt" stores all testing accuracy  
"test_loss.txt" stores all testing loss  
"train_loss.txt" stores all training loss  

# All python files and data files
---"modelnet40_ply_hdf5_2048" (store all raw dataset)  
---"main_train.py" (The main python file, run this)  
---"modeling.py" (Build CNN_CombinePD)  
---"dataloader.py" (Load raw data)  
---"visualization.py" (visualize data)  
---"drawtrain.py" (plot training graphs)  
---"drawtest.py" (plot testing graphs)  





