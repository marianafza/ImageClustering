import os
import math

def stdev(data, xbar=None):
    """Return the square root of the sample variance.

    See ``variance`` for arguments and other details.

    >>> stdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])
    1.0810874155219827

    """
    var = variance(data, xbar)
    try:
        return var.sqrt()
    except AttributeError:
        return math.sqrt(var)
    

def variance(data, xbar=None):
    """Return the sample variance of data.

    data should be an iterable of Real-valued numbers, with at least two
    values. The optional argument xbar, if given, should be the mean of
    the data. If it is missing or None, the mean is automatically calculated.

    """
    if iter(data) is data:
        data = list(data)
    n = len(data)
    if n < 2:
        raise StatisticsError('variance requires at least two data points')
    ss = _ss(data, xbar)
    return ss/(n-1)

def _ss(data, c=None):
    """Return sum of square deviations of sequence data.

    If ``c`` is None, the mean is calculated in one pass, and the deviations
    from the mean are calculated in a second pass. Otherwise, deviations are
    calculated from ``c`` as given. Use the second case with care, as it can
    lead to garbage results.
    """
    if c is None:
        c = mean(data)
    ss = sum((x-c)**2 for x in data)
    # The following sum should mathematically equal zero, but due to rounding
    # error may not.
    ss -= sum((x-c) for x in data)**2/len(data)
    assert not ss < 0, 'negative sum of square deviations: %f' % ss
    return ss

def mean(data):
    """Return the sample arithmetic mean of data. """

    if iter(data) is data:
        data = list(data)
    n = len(data)
    if n < 1:
        raise StatisticsError('mean requires at least one data point')
    return sum(data)/n

def get_imlist(path):        
   return sorted([os.path.join(path,f) for f in os.listdir(path)])

directory_results = 'Results' 

pathResults = get_imlist(directory_results)

final_path = "Final_Results"
if not os.path.exists(final_path):
    os.makedirs(final_path) 

counter = 0
for f_name in pathResults:
    f = open(f_name, "r")
   
    f_name_splitted = f_name.split("_")
    f_name_splitted.pop()
    try: 
        int(f_name_splitted[-1])
        f_name_splitted.pop()
    except:
        True
        
    if "DS_Store" in f_name: 
        continue
    if counter==0:
        rand_indexes = []
        nmi_indexes = []
        lines = f.readlines()
        rand_indexes.append(float(lines[17].split(" ")[-1]))
        nmi_indexes.append(float(lines[18].split(" ")[-1]))
    else:
        if f_name_splitted == f_name_before:
            lines = f.readlines()
            rand_indexes.append(float(lines[17].split(" ")[-1]))
            nmi_indexes.append(float(lines[18].split(" ")[-1]))         
        else:
            avg_rand = sum(rand_indexes)/float(len(rand_indexes))
            avg_nmi = sum(nmi_indexes)/float(len(nmi_indexes))
            std_rand = stdev(rand_indexes)
            std_nmi = stdev(nmi_indexes)      
            
            resulting_file = "FINAL_RESULTS_"+f_name.split("/")[1]
            fw = open(os.path.join(final_path, resulting_file), "w")
            fw.write("Average Rand Index = " + str(avg_rand) + '\n')
            fw.write("Average NMI Index = " + str(avg_nmi) + '\n')
            fw.write("Standard Deviation of Rand Index = " + str(std_rand) + '\n')
            fw.write("Standard Deviation of Average Rand Index = " + str(std_nmi) + '\n')
            fw.close()
            
            rand_indexes = []
            nmi_indexes = []         
            lines = f.readlines()
            rand_indexes.append(float(lines[17].split(" ")[-1]))
            nmi_indexes.append(float(lines[18].split(" ")[-1]))         
          
    f_name_before = f_name_splitted
    counter = counter + 1

f.close()