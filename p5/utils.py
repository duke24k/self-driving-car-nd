import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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


def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256), cspace='HSV'):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    
    if(cspace == 'HSV'):
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=(0,1))
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=(0,1))
    else:
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),  hist_bins=32, hist_range=(0, 256),
                     hog_channel = 'ALL', orient=9, pix_per_cell=8, cell_per_block=2):
    # Create a list to append feature vectors to
    
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range, cspace=cspace)
        # Append the new feature vector to the features list
        
        
                # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            
#            print('hog_features parameters orient %s , pix_per_cell %s , cell_per_block %s' % (orient, pix_per_cell, cell_per_block))
            
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list

    
        features.append(np.concatenate((spatial_features, hist_features,hog_features)))
    # Return list of feature vectors
    return features

def extract_single_features(imgs, cspace='RGB', spatial_size=(32, 32),  hist_bins=16, hist_range=(0, 256),
                     hog_channel = 'ALL', orient=9, pix_per_cell=8, cell_per_block=2):
    
    # Create a list to append feature vectors to
      
    features = []
    my_images = []
 
    # Read in each one by one
    image = imgs
    
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)      
        
#    my_images.append(feature_image) 
            
    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    # Apply color_hist() also with a color space option now
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range, cspace=cspace)
     # Append the new feature vector to the features list
        
        
    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)        
    else:
                
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
     
    # Append the new feature vector to the features list   
    features.append(np.concatenate((spatial_features, hist_features,hog_features)))
   
    # Return list of feature vectors
    
#    return features, my_images
    return features




def slide_window(img_shape, x_start_stop=[None, None], y_start_stop=None,
                    xy_window=None, xy_overlap=(0.5, 0.5)):
    if xy_window is None:
        xy_window = (32, 32)
    if y_start_stop is None:
        y_start_stop = [None, None]
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img_shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img_shape[0]
        
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    
    # Initialize a list to append window positions to
    window_list = []
    
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
            
    # Return the list of windows
    return window_list


def create_windows(pyramid, image_size, xy_overlap=(0.5,0.5)):
    output = []
    for w_size, y_lims in pyramid:
        windows = slide_window(image_size, x_start_stop=[None, None], y_start_stop=y_lims,
                               xy_window=w_size, xy_overlap=xy_overlap)
#                        xy_window=w_size, xy_overlap=(0.5, 0.5))
        output.append(windows)
    return output


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap



def find_cars(img, windows,  svc, image_size= (64,64), cutoff=140, 
                            cspace='RGB', spatial_size=(32,32),  hist_bins=16, 
                            hog_channel='ALL', orient=9, pix_per_cell=8, cell_per_block=2,car_image =False):
    
    
    image = np.copy(img)
    
    pred_dist = []

    bbox_list = []
    jloop = len(windows)
        
    for j in range(jloop):
            
        loop = windows[j]
        
        for i in range(len(loop)):
    
            img_region = image[windows[j][i][0][1]:windows[j][i][1][1],windows[j][i][0][0]:windows[j][i][1][0]]
        
            resized = cv2.resize(img_region, image_size)
        
            my_features = extract_single_features(resized,cspace=cspace, spatial_size=spatial_size,  hist_bins=hist_bins, 
                                                  hog_channel=hog_channel,  orient=orient, pix_per_cell=pix_per_cell,
                                                  cell_per_block=cell_per_block)
                
            y_prediction = svc.decision_function(my_features)
#            print('y_prediction is %s ' % y_prediction)
            pred_dist.append(y_prediction)
           

            if( y_prediction >  cutoff):
                
                                             
                cv2.rectangle(image,(windows[j][i][0][0],windows[j][i][0][1]),(windows[j][i][1][0],windows[j][i][1][1]),(0,0,255),6) 
                bbox_list.append(((windows[j][i][0][0],windows[j][i][0][1]),(windows[j][i][1][0],windows[j][i][1][1])))
    return image, bbox_list, pred_dist

def find_cars_test(img, windows,  X_scaler, clf, image_size= (64,64), cutoff=0.8, 
                            cspace='RGB', spatial_size=(32,32),  hist_bins=16, 
                            hog_channel='ALL', orient=9, pix_per_cell=8, cell_per_block=2,car_image =False):
    
    
    image = np.copy(img)

    bbox_list = []
    jloop = len(windows)
        
    for j in range(jloop):
            
        loop = windows[j]
        
        for i in range(len(loop)):
    
            img_region = image[windows[j][i][0][1]:windows[j][i][1][1],windows[j][i][0][0]:windows[j][i][1][0]]
        
            resized = cv2.resize(img_region, image_size)
        
            my_features = extract_single_features(resized,cspace=cspace, spatial_size=spatial_size,  hist_bins=hist_bins, 
                                                  hog_channel=hog_channel,  orient=orient, pix_per_cell=pix_per_cell,
                                                  cell_per_block=cell_per_block)
            
            scaled_test = X_scaler.transform((np.array(my_features)).reshape(1, -1))
            
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            #test_prediction = svc.predict(scaled_test)
            
#            y_proba = clf.predict_proba(scaled_test)
            y_prediction = clf.predict(scaled_test)
            
#            print("y_proba")
#            print(y_proba)
           
#            if( y_proba[0][1] > cutoff):
            if( y_prediction ==  1):
                                              
                cv2.rectangle(image,(windows[j][i][0][0],windows[j][i][0][1]),(windows[j][i][1][0],windows[j][i][1][1]),(0,0,255),6) 
                bbox_list.append(((windows[j][i][0][0],windows[j][i][0][1]),(windows[j][i][1][0],windows[j][i][1][1])))
    return image, bbox_list


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        

        
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        

        
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


# Debugging purpose 

def make_image_regions(windows, img):

    test_image_list = []  

    for j in range(len(windows)):
    
        loop = windows[j]
        for i in range(len(loop)):
    
            img_region = img[windows[0][i][0][1]:windows[0][i][1][1],windows[0][i][0][0]:windows[0][i][1][0]]
    
            test_image_list.append(img_region)
        
    return test_image_list
"""
def plausible_image_predict(image_list, X_scaler, clf, image_size= (64,64), cutoff=0.8, 
                            cspace='RGB', spatial_size=(32,32),  hist_bins=16, 
                            hog_channel='ALL', orient=9, pix_per_cell=8, cell_per_block=2,car_image =False):
    
    test_predictions = []
    car_images = []

    for i in range(len(image_list)):
        
            # resize image to 
            resized = cv2.resize(image_list[i], image_size)
        
#            my_features, my_image = extract_single_features(resized,cspace=cspace, spatial_size=spatial_size,  hist_bins=hist_bins, 
#                                                            hog_channel=hog_channel)

            my_features = extract_single_features(resized,cspace=cspace, spatial_size=spatial_size,  hist_bins=hist_bins, 
                                                  hog_channel=hog_channel,  orient=orient, pix_per_cell=pix_per_cell,
                                                  cell_per_block=cell_per_block)
            
            #scaled_test = X_scaler.transform((np.array(my_features)).reshape(-1))    
            scaled_test = X_scaler.transform((np.array(my_features)))
            
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            #test_prediction = svc.predict(scaled_test)
            
            y_proba = clf.predict_proba(scaled_test)
    
            test_predictions.append(y_proba)
       
            if( y_proba[0][1] > cutoff and car_image ):
                car_images.append(resized)
        
    if car_images:
        return test_predictions, car_images
    
    else:
        return test_predictions
"""