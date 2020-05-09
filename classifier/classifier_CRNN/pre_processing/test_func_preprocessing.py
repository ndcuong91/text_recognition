import classifier_CRNN.pre_processing.image_preprocessing as ppf
import cv2
import os
import  shutil
import classifier_CRNN.pre_processing.table_border_extraction_fns as tblet
import classifier_CRNN.pre_processing.augment_functions as augment

def test_get_numb_line_img(path):
    list_file = ppf.list_files1(path, 'jpg')
    list_file += ppf.list_files1(path, 'png')
    for pimg in list_file:
        cv2.destroyAllWindows()
        path_img = os.path.join(path, pimg)
        img = cv2.imread(path_img)
        print(ppf.get_numb_line_img(img,True,0.2,0.1, True))

def test_erase_box_cell(path):
    list_file = ppf.list_files1(path, 'jpg')
    list_file += ppf.list_files1(path, 'png')
    for pimg in list_file:
        # cv2.destroyAllWindows()
        path_img = os.path.join(path, pimg)
        img = cv2.imread(path_img)
        cv2.imshow("img",ppf.erase_cell(img,debug= True))
        cv2.waitKey()

def test_extract_table(path):
    list_file = ppf.list_files1(path, 'jpg')
    list_file += ppf.list_files1(path, 'png')
    for pimg in list_file:
        cv2.destroyAllWindows()
        path_img = os.path.join(path, pimg)
        img = cv2.imread(path_img)
        h = img.shape[0]
        print(path_img)
        print('h h2/7 ',h,int(h*2/7))
        cv2.imshow('raw',img)
        hline_list, vline_list = tblet.get_h_and_v_line_bbox_CNX(img)
        list_p_table, hline_list, vline_list = tblet.detect_table(hline_list, vline_list,int(h/20))
        id = -1
        max_s = 0
        count = 0
        for t in list_p_table:
            print("startY ",t.startY)
            if t.startY > h*2/7:
                wt = t.endX - t.startX
                ht = t.endY - t.startY
                print("x y wt ht ",t.startX,t.startY,ht,wt)
                if wt*ht > max_s and  ht < h/2:
                    id = count
                    max_s = wt*ht
            count+=1
        if id != -1:
            list_p_table[id].detect_cells()
            ifb = list_p_table[id].listCells[-1]
            crop_img = ppf.crop_image(img,ifb[0],ifb[1],ifb[2],ifb[3])
            l_if = ppf.get_numb_line_img(crop_img,True,0.2,0.1, False)
            print(l_if)
            cv2.imshow("crop",crop_img)
            c = 0
            for l in l_if:
                x, y, xe, ye,_ = l
                ci = ppf.crop_image(crop_img, x, y, xe, ye)
                if ci.shape[0] > 0 and ci.shape[1] > 0:
                    cv2.imshow("crop_"+str(c),ci)
                c+=1
        cv2.waitKey()

def test_delete_red_sign(path):
    list_file = ppf.list_files1(path, 'jpg')
    list_file += ppf.list_files1(path, 'png')
    for pimg in list_file:
        # cv2.destroyAllWindows()
        path_img = os.path.join(path, pimg)
        img = cv2.imread(path_img)
        ppf.delete_red_sign_in_text_black(img, True)

def test_auto_rotate_image(path):
    list_file = ppf.list_files1(path, 'jpg')
    list_file += ppf.list_files1(path, 'png')
    for pimg in list_file:
        # cv2.destroyAllWindows()
        path_img = os.path.join(path, pimg)
        img = cv2.imread(path_img)
        img_out = ppf.auto_rotation(img, debug= True)
        cv2.imshow('img out ',img_out)
        cv2.waitKey()

if __name__ == "__main__":
    path = 'E:/data_image'
    # get_numb_line_img(cv2.imread(path))
    # test_erase_box_cell(path)
    # test_extract_table(path)
    # path = 'C:/Users/chungnx/Desktop/test_line_text'
    test_auto_rotate_image(path)