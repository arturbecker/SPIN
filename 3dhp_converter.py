import cv2
import numpy as np
import os
import time

import config as cfg

# We want to read the 3dhp dataset videos with OpenCV, and extract the same frames as used in SPIN. The frames used on SPIN are stored on a npz file

mpi_inf_3dhp_train = np.load("data/dataset_extras/mpi_inf_3dhp_train.npz") # load the npz file for the 3dhp dataset
# we will use the "imgname" array inside this npz as the key for what frames to extract from the video

# before we extract the images, let's create the directory tree (we assume to be inside of the 3dhp dataset folder), since the imwrite command can't make dirs on it's own. We start getting the list of folders from the npz:

#this performs a left justification element-wise on the array of frame names, so to keep only the 28 first characters which define the paths, and then by using np.unique we keep only the 128 unique path names
dirs = [cfg.MPI_INF_3DHP_ROOT + s for s in np.unique(np.char.ljust(mpi_inf_3dhp_train['imgname'], 28))]

for path in dirs:
  os.makedirs(path)

# our output images should look like this
# cfg.MPI_INF_3DHP_ROOT/S1/Seq1/imageFrames/video_0/frame_000001.jpg

elapsed_list = []

for path in dirs:
  cap = cv2.VideoCapture(path[:8] + 'imageSequence' + path[19:-1] + '.avi') # opens the video file on "imageSequence/"
  idx = np.flatnonzero(np.char.find(mpi_inf_3dhp_train['imgname'],path)!=-1) # gets the index of all the array elements that pertain to that video on the npz
  
  for frame in idx:
    
    t = time.time()
    
    cap.set(1,int(mpi_inf_3dhp_train['imgname'][frame][-10:-4])-1) # sets the capture to be the frame number of the corresponding filename; the -1 is necessary because the naming scheme is 1-based but the frame count is 0 based.
    success, image = cap.read()
    img_name = mpi_inf_3dhp_train['imgname'][frame]
    cv2.imwrite(img_name, image) # creates the image on the path and name described on img_name
    
    elapsed = time.time() - t
    elapsed_list.append(elapsed)
  print(path)

print(np.sum(elapsed_list))


#for cleaning up
# import shutil
# dirs = np.unique(np.char.ljust(mpi_inf_3dhp_train['imgname'], 20))
# for path in dirs:
#   shutil.rmtree(path)
