import numpy as np
import cv2
pts1 = np.float32([[50,50],[100,100],[0,100]])
pts2 = np.float32([[0,100],[100,100],[0,0]])
M = cv2.getAffineTransform(pts1,pts2)
print("done")