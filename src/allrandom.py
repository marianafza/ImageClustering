import numpy as np
import scipy.cluster.vq 

class AllRandom:
   def __init__(self, _size, _high=5):     
      self.high = _high
      self.size = _size
   
   def obtainCodebook(self, sampled_x, x):

      print 'Obtaining random codebook...'

      high = sampled_x.max()
      low = sampled_x.min()

      t1 = time.time()
      codebook = np.random.randint(10000, size=(self.size,sampled_x.shape[1]))
      codebook = codebook.astype(float)
      codebook = (codebook*(high-low))/10000 + low

      t2 = time.time()
      result = scipy.cluster.vq.vq(x,codebook)
      t3 = time.time()
      projections = result[0]

      print 'time to compute random codebook = ' + str(t2-t1)
      print 'time to compute projections = ' + str(t3-t2)
      return codebook, projections
   
   def writeFileCodebook(self,f):
      f.write("Codebook construction method All Random with parameters: ")
      f.write("Number of clusters = " + str(self.size) + " ")
      f.write("Max value for each feature = " + str(self.high) + " ")
      f.write('\n')       
