import cv2
import os
from os import listdir

def list_files1(directory, extension):
    # print(listdir(directory))
    list_file = []
    for f in listdir(directory):
        if f.endswith('.' + extension):
            list_file.append(f)
    return list_file

path_file = "/data/aicr_data_hw/image_raw/SL/nd.cuong/crop/aicrhw_2020-02-27_10-04"

list_direct = os.listdir(path_file)
for file in list_direct:
    path_ = os.path.join(path_file,file)
    list_GT = list_files1(path_,'txt')
    print("dir:  ",file)
    for gt in list_GT:
        print(gt)
        strl = gt.split('.')
        namefile = strl[0]
        pi = os.path.join(path_,namefile+'.jpg')
        pf = os.path.join(path_,gt)
        filegt = open(pf,"r", encoding="utf-8")
        print("GT: ",filegt.readline())
        img = cv2.imread(pi)
        if img is None:
            print("None img")
        else:
            cv2.imshow("img",img)
            cv2.waitKey()
