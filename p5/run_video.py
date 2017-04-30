import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from skimage.feature import hog
# NOTE: the next import is only valid 
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label

import pickle
import random
from collections import deque
import itertools

from utils import *
from moviepy.editor import VideoFileClip
from IPython.display import HTML


pickle_filename =   "YCrCb0v0"

#pickle_filename =  "HLSALLv0"
#cutoff= 150

with open(pickle_filename, 'rb') as f:
        dataset = pickle.load(f)
        
svc = dataset["svc"]
orient = dataset["orient"]
pix_per_cell = dataset["pix_per_cell"]
cell_per_block = dataset["cell_per_block"]
spatial_size = dataset["spatial_size"]
hist_bins = dataset["hist_bins"]
hog_channel = dataset["hog_channel"]
cspace= dataset["cspace"]
cutoff=dataset["cutoff"]
    
    
print('orient : %d' % orient)
print('pix_per_cell : %s' % pix_per_cell)
print('cell_per_block : %s' %  cell_per_block)
print('spatial_size : %s' %  str(spatial_size))
print('hist_bins : %d' %  hist_bins)
print('svc : %s' %  svc)
print('hog_channel : %s' %  hog_channel)
print('cspace : %s' %  cspace)
print('cutoff : %s' %  cutoff)
    
pyramid = [
            ((42, 42),  [400, 500]),
           ((64, 64),  [400, 500]),
           ((96, 96),  [400, 500]),
           ((128, 128),[450, 578]),
           ((192, 192),[450, None]),
#             ((256, 256),[450, None])
      ]

image_size = (720, 1280)

windows = create_windows(pyramid, image_size,xy_overlap=(0.75, 0.75))


def find_cars_t(img, windows,  svc, image_size= (64,64), cutoff=140, 
                            cspace='RGB', spatial_size=(32,32),  hist_bins=16, 
                            hog_channel='ALL', orient=9, pix_per_cell=8, cell_per_block=2,car_image =False):
    
    
    image = np.copy(img)
    
    pred_dist = []

    bbox_list = []

        
    for window in windows:
    
            img_region = image[window[0][1]:window[1][1],window[0][0]:window[1][0]]
        
            resized = cv2.resize(img_region, image_size)
        
            my_features = extract_single_features(resized,cspace=cspace, spatial_size=spatial_size,  hist_bins=hist_bins, 
                                                  hog_channel=hog_channel,  orient=orient, pix_per_cell=pix_per_cell,
                                                  cell_per_block=cell_per_block)
                
            y_prediction = svc.decision_function(my_features)
#            print('y_prediction is %s ' % y_prediction)
            pred_dist.append(y_prediction)
           

            if( y_prediction >  cutoff):
                
                                             
                cv2.rectangle(image,(window[0][0],window[0][1]),(window[1][0],window[1][1]),(0,0,255),6) 
                bbox_list.append(((window[0][0],window[0][1]),(window[1][0],window[1][1])))
   
    return image, bbox_list, pred_dist
    
 
def pipeline(image, all_windows):
    
    
    box_threshold = 3
    sum_array_threshold = 25
    d = deque(maxlen=15)


       
    t=time.time()  
    
    windows = itertools.chain(*all_windows)
         
    test, box_list, pred_dist = find_cars_t(image, windows, svc, image_size= (64,64), cutoff=cutoff, 
                            cspace=cspace, spatial_size=spatial_size,  hist_bins=hist_bins, 
                            hog_channel=hog_channel,  orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block,car_image=True)
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

# Add heat to each box in box list
    heats = add_heat(heat,box_list)
    
    hot_heat = apply_threshold(heats,box_threshold)
    
#    heat_box.append(hot_heat)

    #    print(hot_heat)
    
    d.append(hot_heat)    
    d_array = np.array(d)
    sum_array = d_array.sum(axis=0)      
    sum_array_thresh = apply_threshold(sum_array,sum_array_threshold)

   
    
#    heatmap= np.clip(hot_heat, 2, 255)

# Find final boxes from heatmap using label function
#    labels = label(heatmap)
    labels = label(sum_array_thresh)
    
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
   
    t2 = time.time()
    print(round(t2-t, 2), 'image processing time...')
    
    return draw_img




   
def process_image(image):
    
    params = {}
    params['windows'] = windows 
    
    all_windows = params['windows']
    
    return pipeline(image, all_windows)
    


def go_video():
    
    print('helo')
     
    video = VideoFileClip("project_video.mp4")
    project_clip = video.fl_image(process_image) #NOTE: this function expects color images!!
    video_output = "base_v03.mp4"
    project_clip.write_videofile(video_output, audio=False)


go_video()
