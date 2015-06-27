import numpy as np
import os
import cv2
import random
import itertools

def randIndex(clusters):

    answers = np.load('answersMatrixNOVO.npy')
    a = 0
    b = 0
    c = 0
    d = 0
    
    N = len(clusters)
    
    count = 0
    for i in range(0,N):
        for j in range(0,N):
            
            if answers[i,j] != 0.0:
                
                count = count + 1
            
                if clusters[i] == clusters[j]:
                    
                    if answers[i,j] == 2:
                        a = a + 1
                    else:
                        c = c + 1
                    
                else:
                    
                    if answers[i,j] == 1:
                        b = b + 1
                    else:
                        d = d + 1
                       
    rand_index = float(a + b)/(a + b + c + d)
    n = a+b+c+d
    aux = float((a+c)*(a+d) + (c+b)*(d+b))
    
    adjusted_rand_index = float(n*(a+b) - aux)/(pow(n,2) - aux)
    
    print 'a =' + str(a)
    print 'b =' + str(b)
    print 'c =' + str(c)
    print 'd =' + str(d)
    print 'Rand Index = ' + str(rand_index)
    print 'Adjusted Rand Index = ' + str(adjusted_rand_index)
    print 'Positive-negative coeficient index =' + str((float(a)/4635 + float(b)/15263)/2)
    
    return adjusted_rand_index

def getARI(clusters, labels):

    random_pairs1 = np.random.randint(low=0, high=1579, size=24932)
    random_pairs2 = np.random.randint(low=0, high=1579, size=24932)
    random_pairs = random_pairs1,random_pairs2
        
    a = 0
    b = 0
    c = 0
    d = 0
    
    N = len(clusters)
    
    count = 0
    for i,j in itertools.izip(random_pairs[0],random_pairs[1]):
            
        if clusters[i] == clusters[j]:
                    
            if labels[i] == labels[j]:
                a = a + 1
            else:
                c = c + 1
                    
        else:
                    
            if labels[i] != labels[j]:
                b = b + 1
            else:
                d = d + 1
                       
    rand_index = float(a + b)/(a + b + c + d)
    n = a+b+c+d
    aux = float((a+c)*(a+d) + (c+b)*(d+b))
    
    adjusted_rand_index = float(n*(a+b) - aux)/(pow(n,2) - aux)
    
    print 'a =' + str(a)
    print 'b =' + str(b)
    print 'c =' + str(c)
    print 'd =' + str(d)
    print 'Rand Index = ' + str(rand_index)
    print 'Adjusted Rand Index = ' + str(adjusted_rand_index)
    print 'Positive-negative coeficient index =' + str((float(a)/4635 + float(b)/15263)/2)
    
    return adjusted_rand_index


#clusters_random = np.random.randint(10, size=1000)
	 
#rand_index = randIndex(clusters_random)



            
                        
                        
                        
                        
                

