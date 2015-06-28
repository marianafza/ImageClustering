import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class KMeans2:
   def __init__(self, _size = 500 ):     
      self.size = _size
   
   def obtainCodebook(self, sampled_x, x):

      print 'Obatining codebook using k-means from scikit-learn...'
      
      sampled_x = np.array(sampled_x)
      sampled_x = sampled_x.astype(float)
      x = np.array(x)
      x = x.astype(float)

      #normalize
      scaled_x_sampled = StandardScaler().fit_transform(sampled_x)
      scaled_x = StandardScaler().fit_transform(x)
      
      kmeans = KMeans(n_clusters=self.size)
      
      kmeans.fit(scaled_x_sampled, y=None)
      
      codebook = kmeans.cluster_centers_
          
      projections = kmeans.predict(scaled_x)
      
      print 'Codebook obtained.'
      
      return codebook, projections
   
   def writeFileCodebook(self,f):
      f.write("Codebook construction method KMeans from Scikit-Learn with parameters: ")
      f.write("Number of clusters = " + str(self.size) + " ")
      f.write('\n')    
