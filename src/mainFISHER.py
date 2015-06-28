#!/usr/bin/python

import sys, getopt
import os
import cv2
import numpy as np
import siftLib
import surfLib
import fastDetector
import starDetector
import randomDetector
import orbLib
import briefDescriptor
import freakDescriptor
import KMeans1
import histogram
import tfidf
import tfidf2
import tfnorm
import tfidfnorm
import time
import datetime
import Dbscan
import Birch
import hierarchicalClustering
import hierarchicalClustScipy
import minibatch
import meanSift
import randomSamplesBook
import allrandom
from sklearn import metrics
import simpleBinarization
import filterMin
import filterMax
import filterMaxMin
import okapi
import sampleKeypoints
import sampleAllKeypoints
#import warnings
import statistics
import xlsxwriter
import communityDetection
import evaluationUsers

import glob
import math
from scipy.stats import multivariate_normal
import random
from sklearn.decomposition import PCA

def dictionary(descriptors, N):
   em = cv2.EM(N)
   em.train(descriptors)
   return np.float32(em.getMat("means")), np.float32(em.getMatVector("covs")), np.float32(em.getMat("weights"))[0]

def likelihood_moment(x, gaussians, weights, k, moment):	
   x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
   probabilities = map(lambda i: weights[i] * gaussians[i], range(0, len(weights)))

   ytk = probabilities[k] / sum(probabilities)
   return x_moment * ytk

def likelihood_statistics(samples, means, covs, weights):
   s0, s1,s2 = {}, {}, {}
   samples = zip(range(0, len(samples)), samples)
   gaussians = {}
   g = [multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True) for k in range(0, len(weights)) ]
   for i,x in samples:
	   gaussians[i] = {k : g[k].pdf(x) for k in range(0, len(weights) ) }

   for k in range(0, len(weights)):
      s0[k] = reduce(lambda a, (i,x): a + likelihood_moment(x, gaussians[i], weights, k, 0), samples, 0)
      s1[k] = reduce(lambda a, (i,x): a + likelihood_moment(x, gaussians[i], weights, k, 1), samples, 0)
      s2[k] = reduce(lambda a, (i,x): a + likelihood_moment(x, gaussians[i], weights, k, 2), samples, 0)
   return s0, s1, s2

def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
   return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, len(w))])

def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
   return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
   return np.float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])

def normalize(fisher_vector):
   v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
   return v / np.sqrt(np.dot(v, v))

def fisher_vector(samples, means, covs, w):
   s0, s1, s2 =  likelihood_statistics(samples, means, covs, w)
   T = samples.shape[0]
   covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
   a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
   b = fisher_vector_means(s0, s1, s2, means, covs, w, T)
   c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
   fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
   fv = normalize(fv)
   return fv

def fisher_features(descriptors,num_kp,gmm):
	
   features = []
   vec_im = []
   j=0
   for i in range(0, len(descriptors)):
	   
      if i==sum(num_kp[:j+1]) and i!=0:
	 print 'Image ' + str(j+1)
	 features.append(fisher_vector(np.array(vec_im),*gmm))
	 vec_im = []
	 vec_im.append(descriptors[i])
	 j=j+1
      else:
	 vec_im.append(descriptors[i])
   
   features.append(fisher_vector(np.array(vec_im),*gmm))	
   features = np.array(features)
   
   print 'Dimensionality of the Fisher Vectors = ' + str(len(features[0]))
   
   return features

def generate_gmm(descriptors_sampled, N):
	
    #Fitting GMM to sampled data	
    start_time = time.time()
    print "Training GMM of size %d.." % N
    means, covs, weights = dictionary(descriptors_sampled, N)
    
    #print 'Means = ' + str(means)
    #print 'Covs = ' + str(covs)
    print 'Weights = ' + str(weights)

    #throw away gaussians with weights that are too small:
    #th = 1.0 / (2*N)
    #means = np.float32([m for k,m in zip(range(0, len(weights)), means) if weights[k] > th])
    #covs = np.float32([m for k,m in zip(range(0, len(weights)), covs) if weights[k] > th])
    #weights = np.float32([m for k,m in zip(range(0, len(weights)), weights) if weights[k] > th])
    #print 'time to train GMM model = ' + str(time.time()-start_time)

    #save("means.gmm", means)
    #save("covs.gmm", covs)
    #save("weights.gmm", weights)
    return means, covs, weights

def get_imlist(path):  
   paths = []
   files = []
   for f in os.listdir(path):
      if 'DS_Store' not in f:
	 paths.append(os.path.join(path,f))
	 files.append(int(f.split('_')[-1].split('.')[0]))

   files,paths = zip(*sorted(zip(files, paths)))
   return paths
   
def run(pathImages,method,keypnt,numpatch,equalnum,imdes,imsample,percentage,codebook,dist,size,fselec,fselec_perc,histnorm,clust,K,pca,nclusters,rep):

   #################################################################
   #
   # Initializations and result file configurations
   #
   #################################################################   
      
   im_dataset_name= pathImages.split('/')[-1]
   
   date_time = datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')
   
   name_results_file = 'FISHER_' + im_dataset_name + '_' + keypnt + '_' + str(numpatch) + '_' + str(equalnum) + '_' + imdes + '_' + imsample + '_' + 'PCA:'+str(pca) + '_' + 'K:' + str(K) + '_' + clust + '_'+ dist + '_' + date_time
   
   #dir_results = 'Results_' + im_dataset_name + '_FISHER_' + date_time
   dir_results = 'Results_FISHER'
   
   if not os.path.exists(dir_results):
      os.makedirs(dir_results)  
      
   file_count = 2
   file_name = os.path.join(dir_results,name_results_file)
   while os.path.exists(file_name + ".txt"):
      file_name = os.path.join(dir_results,name_results_file) + "_" + str(file_count)
      file_count = file_count + 1
   f = open(file_name + ".txt", 'w')
   
   #################################################################
   #
   # Get images
   #
   #################################################################
   
   #pathImages = '/Users/Mariana/mieec/Tese/Development/ImageDatabases/Graz-01_sample'
   
   imList = get_imlist(pathImages)
   
   print 'Number of images read = ' + str(len(imList))
   f.write("Number of images in dataset read: " + str(len(imList)) + "\n")
   
   #################################################################
   #
   # Image description
   #
   #################################################################
   
   #Get detector classes
   det_sift = siftLib.Sift(numpatch, equalnum)
   det_surf = surfLib.Surf(numpatch, equalnum)
   det_fast = fastDetector.Fast(numpatch, equalnum)
   det_star = starDetector.Star(numpatch, equalnum)
   det_orb = orbLib.Orb(numpatch, equalnum)
   det_random = randomDetector.Random(numpatch)
   
   names_detectors = np.array(["SIFT", "SURF", "FAST", "STAR", "ORB", "RANDOM"])
   detectors = np.array([det_sift, det_surf, det_fast, det_star, det_orb, det_random])
   
   #Get the detector passed in the -k argument
   index = np.where(names_detectors==keypnt)[0]
   if index.size > 0:
      detector_to_use = detectors[index[0]]
   else:
      print 'Wrong detector name passed in the -k argument. Options: SIFT, SURF, FAST, STAR, ORB and RANDOM'
      sys.exit()
      
   #FOR RESULTS FILE
   detector_to_use.writeParametersDet(f)
   
   #Get descriptor classes
   des_sift = siftLib.Sift(numpatch, equalnum)
   des_surf = surfLib.Surf(numpatch, equalnum)
   des_orb = orbLib.Orb(numpatch)
   des_brief = briefDescriptor.Brief()
   des_freak = freakDescriptor.Freak()
      
   names_descriptors = np.array(["SIFT", "SURF", "ORB", "BRIEF", "FREAK"])
   descriptors = np.array([des_sift, des_surf, des_orb, des_brief, des_freak])
   
   #Get the detector passed in the -d argument
   index = np.where(names_descriptors==imdes)[0]
   if index.size > 0:
      descriptor_to_use = descriptors[index[0]]
   else:
      print 'Wrong descriptor name passed in the -d argument. Options: SIFT, SURF, ORB, BRIEF and FREAK'
      sys.exit()
      
   #FOR RESULTS FILE
   descriptor_to_use.writeParametersDes(f)   
   
   kp_vector = [] #vector with the keypoints object
   des_vector = [] #vector wih the descriptors (in order to obtain the codebook)
   number_of_kp = [] #vector with the number of keypoints per image
      
   counter = 1
      
   #save current time
   start_time = time.time()   
   
   labels = []
   class_names = []  
   
   #ADDED
   imPaths = []   
   
   #detect the keypoints and compute the sift descriptors for each image
   for im in imList:
      if 'DS_Store' not in im:
         print 'image: ' + str(im) + ' number: ' + str(counter)
         #read image
         img = cv2.imread(im,0)
	 
	 #ADDED
	 imPaths.append(im)   
	 
         #mask in order to avoid keypoints in border of image. size = 40 pixels
         border = 40
         height, width = img.shape
         mask = np.zeros(img.shape, dtype=np.uint8)
         mask[border:height-border,border:width-border] = 1            
         
         #get keypoints from detector
         kp = detector_to_use.detectKp(img,mask)
         
         #get features from descriptor
         des = descriptor_to_use.computeDes(img,kp)
         
         number_of_kp.append(len(kp))
         kp_vector.append(kp)
         if counter==1:
            des_vector = des
         else:
            des_vector = np.concatenate((des_vector,des),axis=0)
         counter += 1   
         
         #for evaluation
         name1 = im.split("/")[-1]
         name = name1.split("_")[0]
                 
         if name in class_names:
            index = class_names.index(name)
            labels.append(index)
         else:
            class_names.append(name)
            index = class_names.index(name)
            labels.append(index)            
            
   #measure the time to compute the description of each image (divide time elapsed by # of images)
   elapsed_time = (time.time() - start_time) / len(imList)
   print 'Time to compute detector and descriptor for each image = ' + str(elapsed_time)   
   
   f.write('Average time to compute detector and descriptor for each image = ' + str(elapsed_time) + '\n')
   
   n_images = len(kp_vector)
   
   average_words = sum(number_of_kp)/float(len(number_of_kp))
   
   print 'Total number of features = ' + str(len(des_vector)) 
   f.write('Total number of features obtained = ' + str(len(des_vector)) + '\n') 
   print 'Average number of keypoints per image = ' + str(average_words) 
   f.write('Average number of keypoints per image = ' + str(average_words) + '\n')
   
   #################################################################
   #
   # Dimentionality reduction
   #
   #################################################################    
   
   #start_time = time.time()
   #print 'Applying PCA...'
   #pca = PCA(n_components=pca)
   #descriptors_reduced = pca.fit(des_vector).transform(des_vector)
   #print 'PCA Applied.'
   #print 'time to apply PCA = ' + str(time.time()-start_time)
   #des_vector = descriptors_reduced   

   #################################################################
   #
   # Image and Keypoint sampling
   #
   ################################################################# 
   
   rand_indexes = []
   nmi_indexes = []
   
   for iteraction in range(0,rep):
      
      print "\nIteraction #" + str(iteraction+1) + '\n'
      f.write("\nIteraction #" + str(iteraction+1) + '\n')
   
      print 'Sampling images and keypoints prior to codebook computation...'
      
      if imsample != "NONE":
         
         sampleKp = sampleKeypoints.SamplingImandKey(n_images, number_of_kp, average_words, percentage)
         sampleallKp = sampleAllKeypoints.SamplingAllKey(percentage)
         
         names_sampling = np.array(["SAMPLEI", "SAMPLEP"])
         sample_method = np.array([sampleKp, sampleallKp])   
         
         #Get the detector passed in the -g argument
         index = np.where(names_sampling==imsample)[0]
         if index.size > 0:
            sampling_to_use = sample_method[index[0]]
         else:
            print 'Wrong sampling method passed in the -g argument. Options: NONE, SAMPLEI, SAMPLEP'
            sys.exit()
            
         #FOR RESULTS FILE
         sampling_to_use.writeFile(f)
      
         des_vector_sampled = sampling_to_use.sampleKeypoints(des_vector)
            
         print 'Total number of features after sampling = ' + str(len(des_vector_sampled))
         f.write('Total number of features after sampling = ' + str(len(des_vector_sampled)) + '\n')
            
         print 'Images and keypoints sampled...'
         
      else:
         print 'No sampling method chosen'
         #FOR RESULTS FILE
         f.write("No method of keypoint sampling chosen. Use all keypoints for codebook construction \n")
         des_vector_sampled = des_vector
      
      #################################################################
      #
      # Fitting GMM to samples
      #
      #################################################################   
      
      print 'Computting GMM...'
      start_time = time.time()
      
      gmm = generate_gmm(np.array(des_vector_sampled), K)  
      print "Time to fit GMM = " + str(time.time() - start_time)
      
      f.write("Time to fit GMM = " + str(time.time() - start_time) + '\n')
      print 'GMM fitted...'
      
      #################################################################
      #
      # Obtaining Fisher Vectors
      #
      ################################################################# 
      
      print 'Computting fisher vectors...'
      start_time = time.time()
      
      fisher_vectors = fisher_features(des_vector, number_of_kp, gmm)
      
      print 'Time to compute fisher vectors = ' + str(time.time() - start_time)  
      f.write('Time to compute fisher vectors = ' + str(time.time() - start_time) + '\n')
      
      #################################################################
      #
      # Clustering of the features
      #
      #################################################################     
      
      #save current time
      start_time = time.time()     
   
      #Get detector classes
      clust_dbscan = Dbscan.Dbscan(dist)
      clust_kmeans = KMeans1.KMeans1([nclusters])
      clust_birch = Birch.Birch(nclusters)
      clust_meanSift = meanSift.MeanSift(nclusters)
      clust_hierar1 = hierarchicalClustering.Hierarchical(nclusters, dist)
      clust_hierar2 = hierarchicalClustScipy.HierarchicalScipy(dist)
      clust_community = communityDetection.CommunityDetection(dist)
      
      names_clustering = np.array(["DBSCAN", "KMEANS", "BIRCH", "MEANSIFT", "HIERAR1", "HIERAR2", "COMM"])
      clustering_algorithm = np.array([clust_dbscan, clust_kmeans, clust_birch, clust_meanSift, clust_hierar1, clust_hierar2, clust_community])
      
      #Get the detector passed in the -a argument
      index = np.where(names_clustering==clust)[0]
      if index.size > 0:
         clustering_to_use = clustering_algorithm[index[0]]
      else:
         print 'Wrong clustering algorithm name passed in the -a argument. Options: DBSCAN, KMEANS, BIRCH, MEANSIFT, HIERAR1, HIERAR2, COMM'
         sys.exit()      
         
      clusters = clustering_to_use.obtainClusters(fisher_vectors)   
      
      #FOR RESULTS FILE
      clustering_to_use.writeFileCluster(f)
      
      elapsed_time = (time.time() - start_time)
      print 'Time to run clustering algorithm = ' + str(elapsed_time) 
      f.write('Time to run clustering algorithm = ' + str(elapsed_time) + '\n')
      
      #ADDED
      nclusters = int(max(clusters)+1)      

      print 'Number of clusters obtained = ' + str(max(clusters)+1)
      f.write('Number of clusters obtained = ' + str(max(clusters)+1) + '\n')
      
      print 'Clusters obtained = ' + str(np.asarray(clusters))
      
      #date_time = datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')
      #np.savetxt('saveClusters_'+date_time+'_.txt', clusters, '%i', ',')
      
      ##################################################################
      ##
      ## Create folder with central images for each cluster
      ##
      ##################################################################  
      
      ##obtain representative images for each cluster
      #central_ims = clust_community.obtainCenteralImages(fisher_vectors, clusters)      
      
      #central_folder = os.path.join(dir_results,'CenterImages')
      #if not os.path.exists(central_folder):
	 #os.makedirs(central_folder)    
      
      #count=0
      #for central_im in central_ims:
	 #filename = os.path.join(central_folder,'Cluster_'+str(count)+'.jpg')
	 #img = cv2.imread(imPaths[central_im],1)
	 #cv2.imwrite(filename, img) 	    
	 #count = count + 1
      
      ##################################################################
      ##
      ## Separate Clusters into folders
      ##
      ##################################################################     
   
      #clusters_folder = os.path.join(dir_results,'Clusters')
      #if not os.path.exists(clusters_folder):
	 #os.makedirs(clusters_folder) 
	 
      #clust_dir = []
      #for iclust in range(0,nclusters):
	 #direc = os.path.join(clusters_folder,'Cluster_'+str(iclust))
	 #if not os.path.exists(direc):
	    #os.makedirs(direc)	 
	 #clust_dir.append(direc)
      
      #for im in range(0,len(imPaths)):
	 #im_name = imPaths[im].split('/')[-1]
	 ##print clust_dir[int(clusters[im])]
	 #filename = os.path.join(clust_dir[int(clusters[im])],im_name)
	 ##print filename
	 #img = cv2.imread(imPaths[im],1)
	 #cv2.imwrite(filename, img) 	
	 	  
      
      #################################################################
      #
      # Evaluation
      #
      #################################################################   
      
      users = 0
      #labels = np.load('IndividualClustersMatrix.npy')
      
      if users == 1:
	 
	 rand_index = evaluationUsers.randIndex(clusters)
	 rand_indexes.append(rand_index)
	 print 'rand_index = ' + str(rand_index)
	 f.write("Rand Index = " + str(rand_index) + "\n")	 
	 
      else:
	 if len(clusters) == len(labels):
   
	    f.write("\nResults\n")
   
	    f.write('Clusters Obtained = ' + str(np.asarray(clusters)))
	    f.write('Labels = ' + str(np.asarray(labels)))
	     
	    rand_index = metrics.adjusted_rand_score(labels, clusters)
	    rand_indexes.append(rand_index)
	    print 'rand_index = ' + str(rand_index)
	    f.write("Rand Index = " + str(rand_index) + "\n")
		 
	    NMI_index = metrics.normalized_mutual_info_score(labels, clusters)
	    nmi_indexes.append(NMI_index)
	    print 'NMI_index = ' + str(NMI_index)   
	    f.write("NMI Index = " + str(NMI_index) + "\n")
   
   if rep > 1:
      f.write("\nFINAL RESULTS\n")
      f.write("Avg Rand Index = " + str(float(sum(rand_indexes))/rep) + "\n")
      f.write("Std Rand Index = " + str(statistics.stdev(rand_indexes)) + "\n")
      if users != 1:
	 f.write("Avg NMI Index = " + str(float(sum(nmi_indexes))/rep) + "\n")
	 f.write("Std NMI Index = " + str(statistics.stdev(nmi_indexes)) + "\n")
   f.close()
