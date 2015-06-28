import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

class Tfidf3:

   def normalizeHist(self, hist, n_words, n_images):  
      print 'Applying tf-idf 3 transformation...'
      
      new_hist = [[0 for x in range(n_words)] for x in range(n_images)] 
      
      hist_np = np.asarray(hist)
      
      transformer = TfidfTransformer(norm=None)
      new_hist = transformer.fit_transform(counts)
                  
      print 'tf-idf applied'
      
      return new_hist
   
   def writeFile(self, f):
      f.write("Histogram normalization type Tf-Idf 3 \n")   

#counts = [[3, 0, 1], [2, 0, 0], [3, 0, 0],[4, 0, 0],[3, 2, 0], [3, 0, 2]]

#tdif = TfIdf()
#norm = tdif.normalizeHist(counts,3,6)

#print norm

#

#print tfidf.toarray()
