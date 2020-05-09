import cv2, os
import numpy as np



def get_list_file_in_folder(dir, ext='png'):
    included_extensions = ['png', 'jpg', 'JPG', 'jpeg', 'PNG']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names


def resizeCMND_old(img_dir, img_name, maxW=1000, maxH=631):
    print (img_name)
    img= cv2.imread(os.path.join(img_dir,img_name))
    CMND_ratio=maxW/maxH
    imgW=img.shape[1]
    imgH=img.shape[0]
    img_ratio=imgW/imgH

    if img_ratio<CMND_ratio:
        newH=maxH
        newW=int(newH*img_ratio)

    else:
        newW=maxW
        newH=int(newW/img_ratio)
    resize=cv2.resize(img,(newW,newH))
    cv2.imwrite(os.path.join('/home/aicr/cuongnd/aicr.core/data/IDcard/CMND_old_1/resize',img_name.replace('.jpeg','.jpg').replace('.png','.jpg').replace('.PNG','.jpg')),resize)

def erode(img_path):
    img = cv2.imread(img_path)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite('/home/aicr/cuongnd/aicr.core/test/bgr_tax_code_2.jpg', erosion)
    cv2.imshow('result', erosion)
    cv2.waitKey(0)


if __name__ == "__main__":
    # data_dir= '../../data/IDcard/CMND_old_1'
    # img_list=get_list_file_in_folder(data_dir)
    #
    # for img in img_list:
    #     resizeCMND_old(data_dir,img)

    erode('/home/aicr/cuongnd/aicr.core/test/bgr_tax_code.jpg')
