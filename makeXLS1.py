import xlsxwriter
import os

def get_imlist(path):        
   return sorted([os.path.join(path,f) for f in os.listdir(path)])

# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('resultsDetDesSize.xlsx')
worksheet = workbook.add_worksheet()

directory_results = 'FinalResults' 

pathResults = get_imlist(directory_results)

worksheet.write(0, 1, "Dataset")
worksheet.write(0, 2, "Detector")
worksheet.write(0, 3, "Number of Keypoints")
worksheet.write(0, 4, "Equal Number of Keypoints")
worksheet.write(0, 5, "Descriptor")
worksheet.write(0, 6, "Sampling Method")
worksheet.write(0, 7, "Codebook Learning Algorithm")
worksheet.write(0, 8, "Distance measure")
worksheet.write(0, 9, "Codebook Size")
worksheet.write(0, 10, "Feature Selection")
worksheet.write(0, 11, "Histogram Normalization")
worksheet.write(0, 12, "Clustering Algorithm")
worksheet.write(0, 13, "Date and Time")
worksheet.write(0, 14, "Rand Index Avg")
worksheet.write(0, 15, "Rand Index Std")
worksheet.write(0, 16, "NMI Index Avg")
worksheet.write(0, 17, "NMI Index Std")

row = 1
for f_name in pathResults:
   f = open(f_name, "r")
   f_name_splitted = f_name.split("_")
   print f_name_splitted
   
   worksheet.write(row,0,"#"+str(row))
   
   for i in range(2,len(f_name_splitted)):
      worksheet.write(row,i-1,f_name_splitted[i])
   
   lines = f.readlines()
   avg_rand = lines[0].split("=")[1]
   worksheet.write(row,i,float(avg_rand))
   avg_nmi = lines[1].split("=")[1]
   worksheet.write(row,i+2,float(avg_nmi))   
   std_rand = lines[2].split("=")[1]
   worksheet.write(row,i+1,float(std_rand))   
   std_nmi = lines[3].split("=")[1]
   worksheet.write(row,i+3,float(std_nmi))   
   
   row = row + 1

workbook.close()