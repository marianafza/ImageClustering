import numpy as np
import scipy.cluster.vq 
import random

class RandomVectors:
   def __init__(self, _size = 500, ):     
      self.size = _size
   
   def obtainCodebook(self, sampled_x, x):

      codebook = []
      for i in range (0, self.size):
         num = random.randint(0, len(x))
         while any((x[num] == xi).all() for xi in codebook):
            num = random.randint(0, len(x))
         codebook.append(x[num])

      codebook = np.array(codebook)

      result = scipy.cluster.vq.vq(x,codebook)
      projections = result[0]     

      return codebook, projections 
   
   def unique_vectors(self, x):
      
      count = 0
      codebook = []
      for i in range (0, len(x)):
         if not any((x[i] == xi).all() for xi in codebook):
            count = count + 1
            codebook.append(x[i])
      
      print count
  
   def writeFileCodebook(self,f):
      f.write("Codebook construction method Random from Feature Vectors with parameters: ")
      f.write("Number of clusters = " + str(self.size) + " ")
      f.write('\n')    
