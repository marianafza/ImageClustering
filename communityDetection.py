import scipy.cluster.hierarchy as scipyH
import scipy.cluster.vq as vq
import numpy as np
import matplotlib.pylab as plt
import math
import community
import networkx as nx
import scipy.spatial.distance as scipyd
import operator

class CommunityDetection:
   def __init__(self, _dist):
      self.dist = _dist
     
   def obtainClusters(self, hist):

      print 'Obatining clusters using Community Detection Clustering...'
   
      #whiten the features (normalization)
      scaled_hist = vq.whiten(hist)
      
      #compute distance matrix of the images using the distance measure chosen
      dist_matrix = scipyd.cdist(scaled_hist, scaled_hist, self.dist)
      
      N = len(dist_matrix[0])
      im_numbers = range(0,N)
      
      #number of neighbors of each image (node) to consider
      n_neighbors = 20
      
      #create a graph
      G = nx.Graph()
          
      #compute the clossest images of each image and the clossest distances
      dist_matrix_copy = dist_matrix
      closest_im = []
      dist_list = []
      #print len(dist_matrix)
      for im in range(0,len(dist_matrix)):
         dist_sorted,closest_im_nums = zip(*sorted(zip(dist_matrix[im].tolist(), im_numbers)))
         #indexes_same = np.where(np.array(dist_sorted)<1)[0]
         #if len(indexes_same)>1:
            #print im
            #closest_im_nums_dist = np.array(closest_im_nums)
            #print closest_im_nums_dist[np.where(np.array(dist_sorted)<1)[0]]
         closest_im.append(closest_im_nums[1:n_neighbors])
         dist_list.append(dist_sorted[1:n_neighbors])
      
      dist_array = np.array(dist_list)
      closest_im = np.array(closest_im)
      
      #compute the median of the distances in order to use it as thresolhold
      median_dist = np.percentile(dist_array,50)
      
      #filter the closseste images using the median distance threshold
      count = 0
      for im in range(0,len(closest_im)):
         ims_passed = closest_im[im][dist_array[im]<median_dist]
         dist_passed = dist_array[im][dist_array[im]<median_dist]
         if len(ims_passed) == 0:
            #if no images are left after filtering, add only a node
            G.add_node(im)
            count = count + 1
         else:
            #add nodes and edges
            #G.add_edges_from([(im,cls) for cls in ims_passed])
            G.add_weighted_edges_from([(im,cls,dist) for cls,dist in zip(ims_passed,dist_passed)])
      
      #compute the best partition for the graph G (Louvain algorithm for community detection)
      partition = community.best_partition(G)
      clusters = partition.values()
      
      #Plot the Graph 
      #nx.draw(G)
      #plt.show()
      

         
      return clusters
   
   def obtainCenteralImages(self, hist, clusters):
      
      print 'Obatining clusters using Community Detection Clustering...'
         
      #whiten the features (normalization)
      scaled_hist = vq.whiten(hist)
      
      #compute distance matrix of the images using the distance measure chosen
      dist_matrix = scipyd.cdist(scaled_hist, scaled_hist, self.dist)
      
      N = len(dist_matrix[0])
      im_numbers = range(0,N)
      
      #number of neighbors of each image (node) to consider
      n_neighbors = 20
      
      #create a graph
      G = nx.Graph()      
      
      #compute the clossest images of each image and the clossest distances
      dist_matrix_copy = dist_matrix
      closest_im = []
      dist_list = []
      #print len(dist_matrix)
      for im in range(0,len(dist_matrix)):
         dist_sorted,closest_im_nums = zip(*sorted(zip(dist_matrix[im].tolist(), im_numbers)))
         closest_im.append(closest_im_nums[1:n_neighbors])
         dist_list.append(dist_sorted[1:n_neighbors])  
         
      dist_array = np.array(dist_list)
      closest_im = np.array(closest_im)  
      
      #compute the median of the distances in order to use it as thresolhold
      median_dist = np.percentile(dist_array,50)       
      
      #create several graphs (the number of clusters)
      G_sep = []
      nclusters = int(max(clusters)+1)
      for i in range(0,nclusters):
         G_sep.append(nx.Graph())        
      
      #compute separate graphs for each community
      #filter the closseste images using the median distance threshold
      count = 0
      for im in range(0,len(closest_im)):
         ims_passed = closest_im[im][dist_array[im]<median_dist]
         dist_passed = dist_array[im][dist_array[im]<median_dist]
         if len(ims_passed) == 0:
            #if no images are left after filtering, add only a node
            G_sep[int(clusters[im])].add_node(im)
            count = count + 1
         else:
            #add nodes and edges
            #G_sep[clusters[im]].add_edges_from([(im,cls) for cls in ims_passed])
            G_sep[int(clusters[im])].add_weighted_edges_from([(im,cls,dist) for cls,dist in zip(ims_passed,dist_passed)])
      
      central_ims = []
      for i in range(0,nclusters):
         centrality = nx.closeness_centrality(G_sep[i])
         central_ims.append(max(centrality.iteritems(), key=operator.itemgetter(1))[0])
      
      central_im_tupple=zip(range(0,N),central_ims)
      #print central_im_tupple  
      
      return central_ims
         
   def writeFileCluster(self,f):
      #f.write("Clustering algorithm Hierarchical from Scipy with parameters: ")
      #f.write("Distance = " + str(self.dist) + " ")
      #f.write("Linkage method = " + str(self.linkage) + " ")
      #f.write("Stop method = " + str(self.stop_method) + " ")
      #f.write("ProportionDist = " + str(self.proportion_dist) + " ")      
      f.write('\n')          

#mat = np.array([[1, 0.5, 0.9],[0.5, 1, -0.5],[0.9, -0.5, 1]])

#h = HierarchicalScipy()
#h.obtainClusters(mat)