#!/usr/bin/python

import sys, getopt
import os
import mainBOF
import mainFISHER
import mainSPM
import mainBOC
import mainBOFC

def main(argv):
   
   #optional argument's values
   rep = 1
   keypnt = None
   numpatch = None
   equalnum = False
   imdes = None
   imsample = None
   percentage = None
   codebook = None
   dist = None
   size = None
   fselec = None
   fselec_perc = None
   histnorm = None
   clust = None
   K = None
   pca = None
   method = None
   regionscolor = None
   sizecolor = None
   levels = None
   nclusters = None
   
   try:
      opts, args = getopt.getopt(argv,"p:k:n:s:d:g:c:t:f:h:a:m:r:K:",["help", "met=","path=","keypnt=", "numpatch=", "equalnum=", "imdes=", "imkeysample=", "codebook=", "size=", "fselec=", "histnorm=","clust=","dist=","rep=","pca=","coloreg=", "sizecolor=", "nclust=", "levels="])
   except getopt.GetoptError:
      print 'ERROR'
      print 'For BOF (Bag-of-Features):'
      print 'python imageClustering.py --met BOF -p <path for images> -k <keypoint detector> -n <number of patches> -s <equal number of patches per image> -d <descriptor> --pca <number of pca components> (optional) -g <Sampling method> -c <codebook method> -t <size of codebook> -f <feature selection method> -h <histogram normalization> -a <clustering algorithm> -m <distance measure> --nclust <number of clusters> (optional) -r <number of repetitions>'
      print 'For BOC (Bag-of-Colors):'
      print 'python imageClustering.py --met BOC -p <path for images> -n <number of patches> -d <descriptor> -g <Sampling method> -c <codebook method> -t <size of codebook> -f <feature selection method> -h <histogram normalization> -a <clustering algorithm> -m <distance measure> --nclust <number of clusters> (optional) -r <number of repetitions>'
      print 'For BOFC (Bag-of-Features-Colors):'
      print 'python imageClustering.py --met BOFC -p <path for images> -k <keypoint detector> -n <number of patches> -s <equal number of patches per image> -d <descriptor> --coloreg <color regions> -g <Sampling method> -c <codebook method> -t <size of codebook> --sizecolor <size of color codebook> -f <feature selection method> -h <histogram normalization> -a <clustering algorithm> -m <distance measure> --nclust <number of clusters> (optional) -r <number of repetitions>'
      print 'For FISHER (Fisher Vectors):'
      print 'python imageClustering.py --met FISHER -p <path for images> -k <keypoint detector> -n <number of patches> -s <equal number of patches per image> -d <descriptor> --pca <number of pca components> -g <Sampling method> -t <number of gaussians> -f <feature selection method> -h <histogram normalization> -a <clustering algorithm> -m <distance measure> --nclust <number of clusters> (optional) -r <number of repetitions>' 
      print 'For SPM (Spatial Pyramid Matching):'
      print 'python imageClustering.py --met SPM -p <path for images> -k <keypoint detector> -n <number of patches> -s <equal number of patches per image> -d <descriptor> --levels <number of levels for the SPM> -g <Sampling method> -f <feature selection method> -h <histogram normalization> -a <clustering algorithm> -m <distance measure> --nclust <number of clusters> (optional) -r <number of repetitions>'      
      sys.exit(2)
   for opt, arg in opts:
      if opt in "help":
         print opt
         print 'For BOF (Bag-of-Features):'
         print 'imageClustering.py --met BOF -p <path for images> -k <keypoint detector> -n <number of patches> -s <equal number of patches per image> -d <descriptor> --pca <number of pca components> (optional) -g <Sampling method> -c <codebook method> -t <size of codebook> -f <feature selection method> -h <histogram normalization> -a <clustering algorithm> -m <distance measure> --nclust <number of clusters> (optional) -r <number of repetitions>'
         print 'For BOC (Bag-of-Colors):'
         print 'imageClustering.py --met BOC -p <path for images> -n <number of patches> -d <descriptor> -g <Sampling method> -c <codebook method> -t <size of codebook> -f <feature selection method> -h <histogram normalization> -a <clustering algorithm> -m <distance measure> --nclust <number of clusters> (optional) -r <number of repetitions>'
         print 'For BOFC (Bag-of-Features-Colors):'
         print 'imageClustering.py --met BOF -p <path for images> -k <keypoint detector> -n <number of patches> -s <equal number of patches per image> -d <descriptor> --coloreg <color regions> -g <Sampling method> -c <codebook method> -t <size of codebook> --sizecolor <size of color codebook> -f <feature selection method> -h <histogram normalization> -a <clustering algorithm> -m <distance measure> --nclust <number of clusters> (optional) -r <number of repetitions>'
         print 'For FISHER (Fisher Vectors):'
         print 'imageClustering.py --met FISHER -p <path for images> -k <keypoint detector> -n <number of patches> -s <equal number of patches per image> -d <descriptor> --pca <number of pca components> -g <Sampling method> -t <number of gaussians> -f <feature selection method> -h <histogram normalization> -a <clustering algorithm> -m <distance measure> --nclust <number of clusters> (optional) -r <number of repetitions>'
         print 'For SPM (Spatial Pyramid Matching):'
         print 'imageClustering.py --met SPM -p <path for images> -k <keypoint detector> -n <number of patches> -s <equal number of patches per image> -d <descriptor> --levels <number of levels for the SPM> -g <Sampling method> -f <feature selection method> -h <histogram normalization> -a <clustering algorithm> -m <distance measure> --nclust <number of clusters> (optional) -r <number of repetitions>'
         sys.exit()
      elif opt in ("-p", "--path"):
         pathImages = arg      
      elif opt == "--met":
         method = arg         
      elif opt in ("-k", "--keypnt"):
         keypnt = arg
      elif opt in ("-n", "--numpatch"):
         numpatch = int(arg)
      elif opt in ("-s", "--equalnum"):
         if arg == 'True':
            equalnum = True
         else:
            equalnum = False
      elif opt in ("-d", "--imdes"):
         imdes = arg
      elif opt in ("-g", "--imkeysample"):
         arg_split = arg.split(":")
         imsample = arg_split[0]
         percentage = float(arg_split[1])
      elif opt in ("-c", "--codebook"):
         codebook = arg           
      elif opt in ("-t", "--size"):
         size = int(arg)     
      elif opt in ("-h", "--histnorm"):
         histnorm = arg
      elif opt in ("-f", "--fselec"):
         arg_split = arg.split(":") 
         fselec = arg_split[0] 
         fselec_perc = [1, 0]
         if fselec == 'FMAX':
            fselec_perc[0] = float(arg_split[1])
         elif fselec == 'FMIN':
            fselec_perc[1] = float(arg_split[1])
         elif fselec == 'FMAXMIN':
            fselec_perc[0] = float(arg_split[1])
            fselec_perc[1] = float(arg_split[2])
      elif opt in ("-a", "--clust"):
         clust = arg
      elif opt in ("-m", "--dist"):
         dist = arg
      elif opt in ("-r", "--rep"):
         rep = int(arg) 
      elif opt in "-K":
         K = int(arg)      
      elif opt in ("--pca"):
         pca = int(arg)
      elif opt in ("--coloreg"):
         regionscolor = int(arg) 
      elif opt in ("--sizecolor"):
         sizecolor = int(arg)     
      elif opt in ("--nclust"):
         nclusters = int(arg) 
      elif opt in ("--levels"):
         levels = int(arg)       
         
   
   if method == "BOF":
      print '\n#############################\n Arguments for BOF (Bag-of-Features) \n#############################\n'
      print 'Image database path: ' + pathImages + '\n'
      print '1.1) Keypoint detector: ' + keypnt
      print '1.2) Number of patches: ' + str(numpatch)
      print '1.3) Same or different number of pacthes per image: ' + str(equalnum)
      print '2) Descriptor: ' + imdes
      if pca!=None:
         print '2.1) Number of pca components: ' + str(pca) 
      print '3) Image and keypoint sampling method: ' + str(imsample)
      print '      Percentage of images/keypoints: ' + str(percentage)
      print '4.1) Codebook construction algorithm: ' + codebook
      print '4.2) Size of codebook: ' + str(size)
      print '5) Feature selection: ' + fselec
      print '      Max threshold for visual words filtering: ' + str(fselec_perc[0]) 
      print '      Min threshold for visual words filtering: ' + str(fselec_perc[1]) 
      print '6) Histogram normalization: ' + histnorm
      print '7.1) Clustering algorithm: ' + clust      
      print '7.2) Distance measure: ' + dist
      if nclusters!=None:
         print '7.3) Number of clusters: ' + str(nclusters)
      print 'Repetitions = ' + str(rep)
      print '\n'
      
   elif method == "FISHER":
      print '\n#############################\n Arguments for FISHER Vectos\n#############################\n'
      print 'Image database path: ' + pathImages + '\n'     
      print '1.1) Keypoint detector: ' + keypnt
      print '1.2) Number of patches: ' + str(numpatch)
      print '1.3) Same or different number of pacthes per image: ' + str(equalnum)
      print '2) Descriptor: ' + imdes
      print '3) Number of Gaussians: ' + str(K)
      print '4) Number of pca components: ' + str(pca)       
      print '5) Image and keypoint sampling method: ' + str(imsample)
      print '      Percentage of images/keypoints: ' + str(percentage)
      print '6.1) Clustering algorithm: ' + clust      
      print '6.2) Distance measure: ' + dist
      if nclusters!=None:
         print '6.3) Number of clusters: ' + str(nclusters)      
      print 'Repetitions = ' + str(rep)
      print '\n'  
      
   elif method == "SPM":
      print '\n#############################\n Arguments for Spatial Pyramid Matching (SPM)\n#############################\n'
      print 'Image database path: ' + pathImages + '\n'
      print '0) Number of Levels: ' + str(levels)
      print '1.1) Keypoint detector: ' + keypnt
      print '1.2) Number of patches: ' + str(numpatch)
      print '1.3) Same or different number of pacthes per image: ' + str(equalnum)
      print '2) Descriptor: ' + imdes
      print '3) Image and keypoint sampling method: ' + str(imsample)
      print '      Percentage of images/keypoints: ' + str(percentage)
      print '4.1) Codebook construction algorithm: ' + codebook
      print '4.2) Size of codebook: ' + str(size)
      print '5) Feature selection: ' + fselec
      print '      Max threshold for visual words filtering: ' + str(fselec_perc[0]) 
      print '      Min threshold for visual words filtering: ' + str(fselec_perc[1]) 
      print '6) Histogram normalization: ' + histnorm
      print '7.1) Clustering algorithm: ' + clust      
      print '7.2) Distance measure: ' + dist
      if nclusters!=None:
         print '7.3) Number of clusters: ' + str(nclusters)      
      print 'Repetitions = ' + str(rep)
      print '\n'
      
   elif method == "BOC":
      print '\n#############################\n Arguments for Bag-of-Colors (BOC)\n#############################\n'
      print 'Image database path: ' + pathImages + '\n'
      print '1.2) Number of patches: ' + str(numpatch)
      print '3) Image and keypoint sampling method: ' + str(imsample)
      print '      Percentage of images/keypoints: ' + str(percentage)
      print '4.1) Codebook construction algorithm: ' + codebook
      print '4.2) Size of codebook: ' + str(size)
      print '5) Feature selection: ' + fselec
      print '      Max threshold for visual words filtering: ' + str(fselec_perc[0]) 
      print '      Min threshold for visual words filtering: ' + str(fselec_perc[1]) 
      print '6) Histogram normalization: ' + histnorm
      print '7.1) Clustering algorithm: ' + clust      
      print '7.2) Distance measure: ' + dist
      if nclusters!=None:
         print '7.3) Number of clusters: ' + str(nclusters)      
      print 'Repetitions = ' + str(rep)
      print '\n'
   
   elif method == "BOFC":
      print '\n#############################\n Arguments for Bag-of-Features fused with Bag-of-Colors\n#############################\n'
      print 'Image database path: ' + pathImages + '\n'
      print '1.1) Keypoint detector: ' + keypnt
      print '1.2) Number of patches: ' + str(numpatch)
      print '1.3) Number of color regions:' + str(regionscolor)      
      print '1.4) Same or different number of pacthes per image: ' + str(equalnum)
      print '2) Descriptor: ' + imdes      
      print '3) Image and keypoint sampling method: ' + str(imsample)
      print '      Percentage of images/keypoints: ' + str(percentage)
      print '4.1) Codebook construction algorithm: ' + codebook
      print '4.2) Size of codebook: ' + str(size)
      print '4.3) Size of color codebook: ' + str(sizecolor)
      print '5) Feature selection: ' + fselec
      print '      Max threshold for visual words filtering: ' + str(fselec_perc[0]) 
      print '      Min threshold for visual words filtering: ' + str(fselec_perc[1]) 
      print '6) Histogram normalization: ' + histnorm
      print '7.1) Clustering algorithm: ' + clust      
      print '7.2) Distance measure: ' + dist
      if nclusters!=None:
         print '7.3) Number of clusters: ' + str(nclusters)      
      print 'Repetitions = ' + str(rep)
      print '\n'   

   else:
      print 'Wrong method selected: ' + method
      sys.exit(2)
   
   return pathImages,method,keypnt,numpatch,equalnum,imdes,imsample,percentage,codebook,dist,size,fselec,fselec_perc,histnorm,clust,K,pca,rep,regionscolor,sizecolor,nclusters,levels
   
def get_imlist(path):        
   return sorted([os.path.join(path,f) for f in os.listdir(path)])
   
if __name__ == "__main__":
   
   #################################################################
   #
   # Get parameters for testing
   #
   #################################################################   
   
   pathImages,method,keypnt,numpatch,equalnum,imdes,imsample,percentage,codebook,dist,size,fselec,fselec_perc,histnorm,clust,K,pca,rep,regionscolor,sizecolor,nclusters,levels = main(sys.argv[1:])
   
   if method == "BOF":
      mainBOF.run(pathImages,method,keypnt,numpatch,equalnum,imdes,imsample,percentage,codebook,dist,size,fselec,fselec_perc,histnorm,clust,K,pca,nclusters,rep)
   elif method == "FISHER":
      mainFISHER.run(pathImages,method,keypnt,numpatch,equalnum,imdes,imsample,percentage,codebook,dist,size,fselec,fselec_perc,histnorm,clust,K,pca,nclusters,rep)
   elif method == "SPM":
      mainSPM.run(pathImages,method,keypnt,numpatch,equalnum,imdes,imsample,percentage,codebook,dist,size,fselec,fselec_perc,histnorm,clust,K,pca,nclusters,rep,levels)
   elif method == "BOC":
      mainBOC.run(pathImages,method,numpatch,imsample,percentage,codebook,dist,size,fselec,fselec_perc,histnorm,clust,nclusters,rep)
   elif method == "BOFC":
      mainBOFC.run(pathImages,method,keypnt,numpatch,equalnum,imdes,imsample,percentage,codebook,dist,size,fselec,fselec_perc,histnorm,clust,K,pca,rep,regionscolor,sizecolor,nclusters)    
   