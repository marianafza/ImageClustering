import numpy as np
import os
import cv2
import random
import itertools

def get_imlist(path):  
    paths = []
    files = []
    for f in os.listdir(path):
	if 'DS_Store' not in f:
	    paths.append(os.path.join(path,f))
	    files.append(int(f.split('_')[-1].split('.')[0]))

    files,paths = zip(*sorted(zip(files, paths)))
    return paths


answers = np.load('answersMatrix.npy')
folder = "/Users/Mariana/mieec/Tese/Development/ImageDatabases/images_trip_processed1000"

N = 1000

imPaths = get_imlist(folder)

count = 0
nclust = 0
clusters = -1*np.ones(1000)
for i in range(0,N):
    for j in range(0,N):  
	
	if answers[i,j] == 2:
	    if clusters[i] != -1 and clusters[j] != -1:
		index = np.where(clusters == clusters[j])
		for ind in index:
		    clusters[ind] = clusters[i]
	    if clusters[i] != -1:
		clusters[j] = clusters[i]
	    elif clusters[j] != -1:
		clusters[i] = clusters[j]                  
	    else:                  
		clusters[i] = nclust
		clusters[j] = nclust
		nclust = nclust + 1
	    count = count + 1
		
#to make the noise points to random numbers instead of -1
nclusters = int(clusters.max()+1)
indexes_noise = np.where(clusters==-1)[0]
random_numbers = random.sample(range(nclusters,nclusters+len(indexes_noise)), len(indexes_noise))
for i,r in itertools.izip(indexes_noise,random_numbers):
    clusters[i] = r  
    
dir_results = 'GroupEvaluationResults'
nclusters = int(clusters.max()+1)
    
clusters_folder = os.path.join(dir_results,'Clusters')
if not os.path.exists(clusters_folder):
    os.makedirs(clusters_folder) 
	     
clust_dir = []
for iclust in range(0,nclusters):
    direc = os.path.join(clusters_folder,'Cluster_'+str(iclust))
    if not os.path.exists(direc):
	os.makedirs(direc)	 
    clust_dir.append(direc)
	  
for im in range(0,len(imPaths)):
    im_name = imPaths[im].split('/')[-1]
    #print clust_dir[int(clusters[im])]
    filename = os.path.join(clust_dir[int(clusters[im])],im_name)
    #print filename
    img = cv2.imread(imPaths[im],1)
    cv2.imwrite(filename, img) 	    

np.save('GroupsClustersMatrix', clusters)