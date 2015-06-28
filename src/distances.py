from sklearn.preprocessing import StandardScaler
import scipy.spatial.distance as scipyd
import numpy as np

def calculateClosest(hist,dist):
    hist = np.array(hist)
    hist = hist.astype(float)
    scaled_hist = StandardScaler().fit_transform(hist)
    dist_matrix = scipyd.cdist(scaled_hist, scaled_hist, dist)
    
    N = len(dist_matrix[0])
    
    dist_matrix_copy = dist_matrix
    closest_im = []
    for im_dists in dist_matrix:
        im_numbers = range(0,N)
        dist_sorted,closest_im_nums = zip(*sorted(zip(im_dists.tolist(), im_numbers)))
        closest_im.append(closest_im_nums[1:6])
    
    return closest_im
    
    