import cv2
import numpy as np
import random
import math
import os
import json
from os import listdir
from datetime import datetime
import shutil

import cv2
import numpy as np

try:
    from pre_processing.table_border_extraction_fns import get_h_and_v_line_bbox_CNX, detect_table
    #from .image_calibration import calib_image
except ImportError:
    from table_border_extraction_fns import get_h_and_v_line_bbox_CNX, detect_table
    #from image_calibration import calib_image


kernel_sharpening = np.array([[0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [-1, -1, -1, -1, 17, -1, -1, -1, -1],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0]])

kernel_blur = np.array([[1 / 9, 1 / 9, 1 / 9],
                        [1 / 9, 1 / 9, 1 / 9],
                        [1 / 9, 1 / 9, 1 / 9]])

kernel_shapping2 = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts, scaleWunderH=np.sqrt(2)):
    # obtain a consistent order of the points and unpack them
    # individually
    pts = np.array(pts, dtype="float32")
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_AREA)
    # resize A4
    warped = cv2.resize(warped, (maxWidth, int(maxWidth * scaleWunderH)), interpolation=cv2.INTER_NEAREST)

    # return the warped image
    return warped


def perspective_transform(image, pts, maxWidth=856, maxHeight=540):
    pts = np.array(pts, dtype="float32")
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_NEAREST)
    return warped


def rotate_image_angle(img, angle, pixel_erase=1):
    shape_ = img.shape
    h_org = shape_[0]
    w_org = shape_[1]
    Mat_rotation = cv2.getRotationMatrix2D((w_org / 2, h_org / 2), 360 - angle, 1)
    # rotate image
    abs_cos = abs(Mat_rotation[0, 0])
    abs_sin = abs(Mat_rotation[0, 1])

    bound_w = int(h_org * abs_sin + w_org * abs_cos)
    bound_h = int(h_org * abs_cos + w_org * abs_sin)

    Mat_rotation[0, 2] += bound_w / 2 - w_org / 2
    Mat_rotation[1, 2] += bound_h / 2 - h_org / 2

    img_result = cv2.warpAffine(img, Mat_rotation, (bound_w, bound_h))
    blank_image = np.zeros(shape=[h_org, w_org, 1], dtype=np.uint8)
    blank_image = 255 - blank_image
    img_result_blank = cv2.warpAffine(blank_image, Mat_rotation, (bound_w, bound_h))
    img_result_blank = 255 - img_result_blank
    img_result = erase_img(img_result, img_result_blank, None, 1)
    return img_result


def flip_image(src, flip):
    # flip 0 flip up down, 1 flip left right, -1 flip up down left right
    result = None
    result = cv2.flip(src, flip)
    return result


def invert_image(src, threshold = 41):
    h, w, c = src.shape
    gray = None
    if (c == 3):
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src
    # convert to binary
    adaptivethreshold_applied_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                          cv2.THRESH_BINARY, threshold, 7)
    invert_img = 255 - adaptivethreshold_applied_img
    return invert_img


def erase_img(img, temp , rect , pixel_cal=5, method=cv2.INPAINT_TELEA):
    # format  of rect [x1, y1, x2, y2]
    # (x1, y1) coordinate top left, (x2, y2) coordinate bottom right
    dst = None
    if temp is None:
        h = img.shape[0]
        w = img.shape[1]
        x1, y1, x2, y2 = rect
        blank_image = np.zeros(shape=[h, w, 1], dtype=np.uint8)
        for r in range(y1, (y2 + 1)):
            for c in range(x1, (x2 + 1)):
                if r < h and c < w:
                    blank_image[r, c] = 255
        # mask = cv2.threshold(blank_image,100,255,cv2.THRESH_BINARY)
        dst = cv2.inpaint(img, blank_image, pixel_cal, method)
    else:
        dst = cv2.inpaint(img, temp, pixel_cal, method)
    return dst


def auto_rotation(img, expand_angle=3, outputangle = False,debug = False):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.medianBlur(gray_img,3)
    h_img, w_img = gray_img.shape
    scale = 1500./min(h_img,w_img)
    gray_img = cv2.resize(gray_img,None,fx= scale , fy= scale)
    h_img, w_img = gray_img.shape
    edges = cv2.Canny(gray_img, 60, 150, apertureSize=3)
    blank_image = np.zeros(shape=[h_img, w_img, 1], dtype=np.uint8)
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    list_bb = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # if w > 5 and h >5:
        list_bb.append([x, y, x + w, y + h])
    img_cp = img.copy()
    for b in list_bb:
        centerx = int((b[2] - b[0]) / 2 + b[0])
        centery = int((b[3] - b[1]) / 2 + b[1])
        cv2.circle(blank_image, (centerx, centery), 1, (255, 255, 255), -1)
    minLineLen = min(h_img,w_img)/8
    if debug == True:
        print(min(h_img,w_img))
        img_cp2 = gray_img.copy()
        print('min ', minLineLen)
    lines = cv2.HoughLinesP(blank_image, 3, np.pi / 180, threshold=90, minLineLength=minLineLen, maxLineGap=35)
    list_angle = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        angle = (math.atan2((y1 - y2), (x1 - x2)) * 180) / math.pi
        list_angle.append(angle)
        if debug == True:
            cv2.line(img_cp2,(x1, y1),(x2, y2),(255,0,0), 1)
    list_cluster_angle = []
    max_candidate = 0
    angle_rotate = 0
    for ag in list_angle:
        hascluster = False
        for cluster in list_cluster_angle:
            if abs(ag - cluster[0]) < expand_angle:
                cluster[0] = (cluster[0] * cluster[1] + ag) / (cluster[1] + 1)
                cluster[1] += 1
                if cluster[1] > max_candidate:
                    max_candidate = cluster[1]
                    angle_rotate = cluster[0]
                hascluster = True
                break
        if hascluster == False:
            new_cluster = [ag, 1]
            list_cluster_angle.append(new_cluster)
    convert_angle_opencv = 180 - angle_rotate
    if debug == True:
        cv2.imshow('blank img', blank_image)
        cv2.imshow('img_cp', img_cp2)
        cv2.imshow('img ed',edges)
        print('angle ',convert_angle_opencv)
    if outputangle == True:
        return  convert_angle_opencv
    img_cp = rotate_image_angle(img_cp, convert_angle_opencv)
    return img_cp


def crop_image(img, x1, y1, x2, y2, offsetx=0, offsety=0):
    return img[int(y1 + offsety):int(y2 + offsety), int(x1 + offsetx):int(x2 + offsetx)]


def get_coor_match_template(img, template, type_img=0, numb_scale=50):
    # type 0 rgb, 1 gray, 3 binary
    img_process = img.copy()
    temp_process = template.copy()
    if type_img == 1:
        img_process = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        temp_process = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    elif type_img == 3:
        img_process = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        temp_process = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        img_process = cv2.adaptiveThreshold(img_process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 41, 7)
        temp_process = cv2.adaptiveThreshold(temp_process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 41, 7)
    h_temp = template.shape[0]
    w_temp = template.shape[1]
    low_scale = max(template.shape[1] / img.shape[1], template.shape[0] / img.shape[0])
    lscales = np.linspace(1, low_scale, num=numb_scale)
    found = None
    for scale in lscales:
        imrs = cv2.resize(img_process, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h_imrs = imrs.shape[0]
        w_imrs = imrs.shape[1]
        if h_imrs < h_temp or w_imrs < w_temp:
            break
        res = cv2.matchTemplate(imrs, temp_process, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if found is None or max_val > found[0]:
            found = [max_val, max_loc[0], max_loc[1], scale]
    print("max val : ", max_val)
    top_left = [found[1], found[2]]
    top_left[0] /= found[3]
    top_left[1] /= found[3]
    w = w_temp / found[3]
    h = h_temp / found[3]
    coor_temp1 = [int(top_left[0]), int(top_left[1]), int(top_left[0] + w), int(top_left[1] + h)]
    return coor_temp1


class clTemplate:
    def __init__(self):
        self.height = 0
        self.width = 0
        self.name_template = ''
        self.listBoxtemplate = []
        self.listBoxinfor = []


def auto_extract_info(img, path_config_file, threshold=0.75):
    list_template = []
    list_filebackground = os.listdir(path_config_file)
    print(list_filebackground)
    if len(list_filebackground) == 0:
        print("template files empty")
        return
    else:
        for i in list_filebackground:
            template = clTemplate()
            path_f = os.path.join(path_config_file, i)
            with open(path_f, 'r+') as readf:
                count = 0
                for line in readf:
                    count += 1
                    if count == 1:
                        template.name_template = line
                    else:
                        list_inf = line.split()
                        if len(list_inf) == 7:
                            bb_t = [int(list_inf[0]), int(list_inf[1]), int(list_inf[2]), \
                                    int(list_inf[3]), float(list_inf[4]), float(list_inf[5]), float(list_inf[6])]
                            template.listBoxtemplate.append(bb_t)
                        elif len(list_inf) == 4:
                            bb_i = [int(list_inf[0]), int(list_inf[1]), int(list_inf[2]), int(list_inf[3])]
                            template.listBoxinfor.append(bb_i)
                        elif len(list_inf) == 2:
                            template.height = int(list_inf[0])
                            template.width = int(list_inf[1])
            list_template.append(template)
    img_bl = auto_rotation(img)
    im_bl_gray = cv2.cvtColor(img_bl, cv2.COLOR_BGR2GRAY)
    h_n, w_n = im_bl_gray.shape
    im_edges = cv2.Canny(im_bl_gray, 50, 150, apertureSize=3)
    major = cv2.__version__.split('.')[0]
    center_img = [int(w_n / 2), int(h_n / 2)]
    if major == '3':
        _, contours, _ = cv2.findContours(im_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(im_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    list_bb_img = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w > 15 and h > 15):
            center_bb = [int(x + w / 2), int(y + h / 2)]
            # ratio_center = (center_tmp[0] - center_bb[0]) / (center_tmp[1] - center_bb[1])
            ratio_w = (center_img[0] - center_bb[0]) / w
            ratio_h = (center_img[1] - center_bb[1]) / h
            list_bb_img.append([x, y, w, h, w / h, ratio_w, ratio_h])
    list_matching_temp = []
    for templ in list_template:
        list_bb_tmp = templ.listBoxtemplate
        if (len(list_bb_img) / len(list_bb_tmp)) < threshold:
            list_matching_temp.append(0)
            continue
        outlier_boxes = []
        for tmp_bb in list_bb_tmp:
            outlier = True
            for img_bb in list_bb_img:
                if tmp_bb[4] != 0 and tmp_bb[5] != 0 and tmp_bb[6] != 0:
                    if (img_bb[4] / tmp_bb[4]) > 0.8 and (img_bb[4] / tmp_bb[4]) < 1.2 \
                            and (img_bb[5] / tmp_bb[5]) > 0.8 and (img_bb[5] / tmp_bb[5]) < 1.2 \
                            and (img_bb[6] / tmp_bb[6]) > 0.8 and (img_bb[6] / tmp_bb[6]) < 1.2:
                        outlier = False
                        break
            if outlier == True:
                outlier_boxes.append(tmp_bb)
        list_matching_temp.append(len(outlier_boxes) / len(list_bb_tmp))
    confi_match = min(list_matching_temp)
    print("conf: ", confi_match)
    if (1 - confi_match) < threshold:
        print("not match")
        return
    id_temp_match = list_matching_temp.index(confi_match)
    template_m = list_template[id_temp_match]
    ratioy = h_n / template_m.height
    ratiox = w_n / template_m.width
    count = 0
    for if_box in template_m.listBoxinfor:
        count += 1
        bx = int(if_box[0] * ratiox)
        by = int(if_box[1] * ratioy)
        ex = int(if_box[2] * ratiox)
        ey = int(if_box[3] * ratioy)
        info_img = crop_image(img_bl, bx, by, ex, ey)
        cv2.imwrite("crop_info" + str(count) + ".jpg", info_img)
        cv2.waitKey()


class clTemplate_demo:
    def __init__(self):
        self.height = 0
        self.name_template = ''
        self.width = 0
        self.imageSource = 0
        self.category = 0
        self.listBoxinfor = []


class clImageInfor_demo:
    def __init__(self):
        self.id = 0 #unique
        self.data = []  #image data for recognize
        self.name = ""  #user define (UI)
        # self.label = "" #display in UI . no need
        self.type = ""  # data rule. e.g: address, date, ...
        # self.nameTemplate = "" #for
        # self.location = []  #bounding box location
        self.value = ""  # value for ocr
        self.value_nlp = ""
        self.confidence = {'ocr':0,'nlp':0,'total':0}
        self.data_type = "" # e.g: text, number, date, image, mark for UI

class clMark_demo:
    def __init__(self):
        self.name = None
        self.id = 0
        self.type = ""
        self.location = []
        self.ischecked = False

class clGroupMark_demo:
    def __init__(self):
        self.id = 0
        self.name = None
        self.label = ""
        self.type = ""
        self.list_mark_info = []
        self.data_type = ""

def background_subtract(image, bgr_path=''):
    # image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # background=cv2.imread(bgr_path, 0)
    background = cv2.imread(bgr_path)
    try:
        result = cv2.subtract(background, image)
        result_inv = cv2.bitwise_not(result)
        # cv2.imshow('result',result_inv)
        # cv2.waitKey(0)
        return result_inv
    except:
        return image


lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([30, 255, 255])

# upper red
lower_red2 = np.array([165, 50, 50])
upper_red2 = np.array([180, 255, 255])

lower_black = np.array([
    0, 0, 0])
upper_black = np.array([180, 255, 180])

def filter_color_image(img, lower_hsv, uper_hsv, binaryimage=True):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, lower_hsv, uper_hsv)
    if binaryimage == True:
        return mask
    else:
        res = cv2.bitwise_and(img, img, mask=mask)
        return res

def delete_red_sign_in_text_black(img, debug = False):
    img1 = filter_color_image(img, lower_red1, upper_red1)
    img2 = filter_color_image(img, lower_red2, upper_red2)
    img3 = filter_color_image(img, lower_black, upper_black)
    img1 += img2
    imgrs = cv2.bitwise_not(img1, mask=img3)
    if debug == True:
        cv2.imshow('mask red',img1)
        cv2.imshow('mask black',img3)
        cv2.imshow('result',(255 - imgrs))
        cv2.waitKey()
    return 255 - imgrs

def extract_for_demo_json(img, temp_json, image_directory = None, eraseline = False, subtract_bgr = False,
                          gen_bgr=False, show_text = False, erasecell = True, splitline = True):
    img_save_bl = img.copy()
    data_dict = temp_json
    template = clTemplate_demo()
    template.height = data_dict['image_size']['height']
    template.width  = data_dict['image_size']['width']
    template.name_template = data_dict['name']
    template.type =  data_dict['type']
    template.category = data_dict['category']
    template.imageSource = data_dict['image_source']
    dict_fields = data_dict['fields']
    font = cv2.FONT_HERSHEY_SIMPLEX
    h_n = img.shape[0]
    w_n = img.shape[1]
    ratioy = h_n / template.height
    ratiox = w_n / template.width
    dict_class_img_info = {}
    list_group_mark = []
    max_wh_ratio = 1
    for df in dict_fields:
        if df['type'] != 'mark':
            classimginf = clImageInfor_demo()
            classimginf.id = df['id']
            classimginf.name = df['name']
            # classimginf.label = df['label']
            classimginf.type = df['type']
            classimginf.data_type = df['data_type'] if 'data_type' in df else 'text'
            classimginf.confidence["ocr"] = 0.0
            classimginf.confidence["nlp"] = 0.0
            classimginf.confidence["total"] = 0.0
            by = int(df['position']['top'])
            bx = int(df['position']['left'])
            ex = bx + int(df['size']['width'])
            ey = by + int(df['size']['height'])
            wh_ratio = float(ex - bx) / float(ey - by)
            # print('wh_ratio',wh_ratio)
            if wh_ratio > max_wh_ratio:
                max_wh_ratio = wh_ratio
            bx = int(bx * ratiox)
            ex = int(ex * ratiox)
            by = int(by * ratioy)
            ey = int(ey * ratioy)
            # classimginf.location = [bx, by, ex, ey]
            offsetx, offsety = 1, 1
            if gen_bgr:
                offsetx, offsety = 0, 0
            info_img = crop_image(img, bx, by, ex, ey, offsety=offsety, offsetx=offsetx)
            if classimginf.data_type != 'image':
                if gen_bgr:
                    cv2.imwrite('classifier_CRNN/form/background_VIB/' + str(classimginf.id) + '.jpg', info_img)
                if subtract_bgr:
                    info_img = background_subtract(info_img, bgr_path='classifier_CRNN/form/background_VIB/' + str(classimginf.id) + '.jpg')
                if eraseline == True:
                    rs = removeLines(info_img, expand_bb=3, pixel_erase=2)
                    if rs is not None:
                        info_img = rs
                if erasecell == True:
                    info_img = erase_cell(info_img)
                if splitline == True:
                    list_coor = get_numb_line_img(info_img, True, 0.2, 0.2)
                    if len(list_coor) > 0:
                        for l in list_coor:
                            cr_img = crop_image(info_img, l[0], l[1], l[2], l[3])
                            classimginf.data.append(cr_img)
                    else:
                        classimginf.data.append(info_img)
                else:
                    classimginf.data.append(info_img)
            else:
                classimginf.data.append(info_img)

            print('---------------{0}'.format(classimginf.data_type))
            if image_directory is not None and classimginf.data_type == 'image':
                imgcrp = crop_image(img,bx,by,ex,ey)
                path_save_crop_directory = os.path.join(image_directory, 'crops/')
                path_save_crop_file = os.path.join(path_save_crop_directory, str(classimginf.id) + '.jpg')

                if not os.path.exists(path_save_crop_directory):
                    os.umask(0)
                    os.makedirs(path_save_crop_directory)

                cv2.imwrite(path_save_crop_file, imgcrp)
                classimginf.value = path_save_crop_file

            dict_class_img_info[classimginf.name] = classimginf
            cv2.rectangle(img_save_bl, (bx, by), (ex, ey), (0, 0, 255), 4)
            if show_text:
                cv2.putText(img_save_bl, classimginf.name, (bx - 100, by + 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            classimginf.location = [bx, by, ex, ey]
        else:
            group_mark = clGroupMark_demo()
            group_mark.type = df['type']
            group_mark.name = df['name']
            # group_mark.label = df['label']
            group_mark.id = df['id']
            group_mark.data_type = df['data_type']
            dict_option = df['options']
            for dopt in dict_option:
                clmark = clMark_demo()
                clmark.name = dopt['name']
                # clmark.label = dopt['label']
                clmark.type = dopt['type']
                clmark.id   = dopt['id']
                by = int(dopt['position']['top'])
                bx = int(dopt['position']['left'])
                ex = bx + int(dopt['size']['width'])
                ey = by + int(dopt['size']['height'])
                bx = int(bx * ratiox)
                ex = int(ex * ratiox)
                by = int(by * ratioy)
                ey = int(ey * ratioy)
                # clmark.location = [bx, by, ex, ey]
                info_img = crop_image(img, bx, by, ex, ey, offsety=offsety, offsetx=offsetx)
                # cv2.imshow("mark",info_img)
                # cv2.waitKey()
                checked = check_mark(info_img)
                clmark.ischecked = checked
                group_mark.list_mark_info.append(clmark)
                cv2.rectangle(img_save_bl, (bx, by), (ex, ey), (0, 0, 255), 4)
                if show_text:
                    cv2.putText(img_save_bl, clmark.name, (bx - 100, by + 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            list_group_mark.append(group_mark)
    if image_directory is not None:
        save_path_rs = os.path.join(image_directory, 'template.jpg')
        cv2.imwrite(save_path_rs, img_save_bl)

    return dict_class_img_info, max_wh_ratio , list_group_mark



def extract_for_demo(listimg, path_config_file, save_to_database=False, eraseline=False, subtract_bgr=True, gen_bgr=False):
    if len(listimg) == 0:
        print("input image empty")
        return
    template = clTemplate_demo()
    max_wh_ratio = 1
    with open(path_config_file, 'r+') as readf:
        count = 0
        for line in readf:
            count += 1
            if count == 1:
                template.name_template = line
            else:
                list_inf = line.split()
                if len(list_inf) == 5:
                    bb_i = [list_inf[0], int(list_inf[1]), int(list_inf[2]),
                            int(list_inf[1]) + int(list_inf[3]), int(list_inf[2]) + int(list_inf[4])]
                    wh_ratio = float(list_inf[3]) / float(list_inf[4])
                    if wh_ratio > max_wh_ratio:
                        max_wh_ratio = wh_ratio
                    template.listBoxinfor.append(bb_i)
                elif len(list_inf) == 2:
                    template.height = int(list_inf[0])
                    template.width = int(list_inf[1])
    #print('max wh ratio',max_wh_ratio)
    count_img = 0
    list_class_img_info = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for img in listimg:
        img_bl = img
        img_save_bl = img_bl.copy()
        h_n = img_bl.shape[0]
        w_n = img_bl.shape[1]
        ratioy = h_n / template.height
        ratiox = w_n / template.width
        for if_box in template.listBoxinfor:
            classimginf = clImageInfor_demo()
            namef, typef, labelf = if_box[0].split('/')
            classimginf.name = namef
            classimginf.type = typef
            labelf = labelf.replace('.',' ')
            classimginf.label = labelf
            # classimginf.nameTemplate = template.name_template
            count_img += 1
            bx = int(if_box[1] * ratiox)
            by = int(if_box[2] * ratioy)
            ex = int(if_box[3] * ratiox)
            ey = int(if_box[4] * ratioy)
            if save_to_database:
                cv2.rectangle(img_save_bl, (bx, by), (ex, ey), (0, 0, 255), 4)
                cv2.putText(img_save_bl, if_box[1], (bx - 100, by + 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            classimginf.location = [bx, by, ex, ey]
            offsetx, offsety = 1, 1
            if gen_bgr:
                offsetx, offsety = 0, 0
            info_img = crop_image(img_bl, bx, by, ex, ey, offsety=offsety, offsetx=offsetx)
            if gen_bgr:
                cv2.imwrite('form/background_VIB/' + if_box[0] + '.jpg', info_img)
            if subtract_bgr:
                info_img = background_subtract(info_img, bgr_path='form/background_VIB/' + if_box[0] + '.jpg')
            if eraseline == True:
                rs = removeLines(info_img, expand_bb=3, pixel_erase=2)
                if rs is not None:
                    info_img = rs
            classimginf.data = info_img
            list_class_img_info.append(classimginf)
        if save_to_database:
            cv2.imwrite("/data/data_imageVIB/result.jpg", img_save_bl)
    return list_class_img_info, max_wh_ratio

def get_numb_line_img(img, scale_img = True , expand_h = 0, expand_w = 0, debug = False):
    h_img = img.shape[0]
    w_img = img.shape[1]
    fscale = 1
    if h_img < 150 and scale_img == True:
        fscale = 150/h_img
        imgrs = cv2.resize(img,None,None,fx=fscale, fy=fscale,interpolation=cv2.INTER_CUBIC)
    else:
        imgrs = img
    imgv = cv2.cvtColor(imgrs,cv2.COLOR_BGR2GRAY)
    edg2 = invert_image(imgrs)
    if debug == True:
        cv2.imshow("imgv ", imgv)
        cv2.imshow("edg",edg2)
        img3 = img.copy()
        img2 = imgrs.copy()
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, he = cv2.findContours(edg2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    else:
        contours, he = cv2.findContours(edg2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    list_cluster_line = []
    list_boundingRect = []
    h_max = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w/h < 4 and h/w < 4:
            if h > h_max:
                h_max = h
            list_boundingRect.append([x, y, w, h])
    list_filter_bb = []
    for b in list_boundingRect:
        if b[3] >= h_max/5  and b[2] > 0:
            xend = b[0] + b[2]
            yend = b[1] + b[3]
            list_filter_bb.append([b[0], b[1], xend, yend])
            if debug == True:
                cv2.rectangle(img2, (b[0], b[1]), (xend, yend), (255, 0, 0), 1)
    list_filter_bb.sort(key=lambda x: (x[1], x[0]))
    for r in list_filter_bb:
        check_in_cluster = False
        x, y, xend, yend = r
        if len(list_cluster_line) > 0:
            count = -1
            max_area = 0
            id_cls = -1
            for l in list_cluster_line:
                count += 1
                xb = l[0]
                yb = l[1]
                xe = l[2]
                ye = l[3]
                area_overlapse = min(yend, ye) - max(yb, y)
                if area_overlapse < 0:
                    continue
                elif (area_overlapse /(ye - yb) > 0.4) or (area_overlapse/(yend - y) > 0.4):
                    if max_area < area_overlapse:
                        max_area = area_overlapse
                        id_cls = count
                        check_in_cluster = True
            if id_cls != -1:
                list_cluster_line[id_cls][0] = min(x, list_cluster_line[id_cls][0])
                list_cluster_line[id_cls][1] = min(y, list_cluster_line[id_cls][1])
                list_cluster_line[id_cls][2] = max(xend, list_cluster_line[id_cls][2])
                list_cluster_line[id_cls][3] = max(yend, list_cluster_line[id_cls][3])
                list_cluster_line[id_cls][4] += 1
        if len(list_cluster_line) == 0 or check_in_cluster == False:
            list_cluster_line.append([x, y, xend, yend, 1])

    filter_list = []
    count = 0
    for l in list_cluster_line:
        w = l[2] - l[0]
        h = l[3] - l[1]
        if l[4] > 1 and w/h > 3 and h > h_max/2:
            l[0] = int(l[0] / fscale)
            l[1] = int(l[1] / fscale)
            l[2] = int(l[2] / fscale)
            l[3] = int(l[3] / fscale)
            w_ = l[2] - l[0]
            h_ = l[3] - l[1]
            exp_h = int(h_ * expand_h)
            exp_w = int(w_ * expand_w)
            l[0] = max(0, l[0] - exp_w)
            l[1] = max(0, l[1] - exp_h)
            l[2] = min(w_img, l[2] + exp_w)
            l[3] = min(h_img, l[3] + exp_h)
            filter_list.append(l)
            if debug == True:
                cv2.rectangle(img3, (l[0], l[1]), (l[2], l[3]), (0, 255, 0))
                imcr = crop_image(img3,l[0],l[1],l[2],l[3])
                if imcr.shape[0] > 0 and imcr.shape[1]>0:
                    cv2.imshow('crop'+str(count),imcr)
                count+=1
    if debug == True:
        print(list_cluster_line)
        print(filter_list)
        cv2.imshow("img3",img3)
        cv2.imshow("img",img2)
        cv2.waitKey()
    #format output list[[xtl, ytl, xbtr, ybtr, count]]
    filter_list.sort(key=lambda x: (x[1]))
    return filter_list


def erase_cell(img,threshold = 0.2,debug = False):
    h_img = img.shape[0]
    w_img = img.shape[1]
    list_h_ct = []
    list_v_ct = []
    threshold_dis = max(5,int(threshold * h_img))
    blank_image = np.zeros(shape=[h_img, w_img, 1], dtype=np.uint8)
    invimg = invert_image(img)
    if debug == True:
        print(threshold_dis)
        cv2.imshow("imiv ",invimg)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (threshold_dis, 1))
    horizontal_extraction_img = cv2.morphologyEx(invimg, cv2.MORPH_ERODE, kernel, iterations=3)

    horizontal_extraction_buffup_iter_5_img = cv2.morphologyEx(horizontal_extraction_img, cv2.MORPH_DILATE, kernel,
                                                               iterations=3)
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, hcontours, _ = cv2.findContours(horizontal_extraction_buffup_iter_5_img, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    else:
        hcontours, _ = cv2.findContours(horizontal_extraction_buffup_iter_5_img, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_NONE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, threshold_dis))
    for c in hcontours:
        x, y, w, h = cv2.boundingRect(c)
        if w > w_img/3:
            list_h_ct.append([x, y, x + w, y + h])
    if len(list_h_ct) < 2:
        return img
    print(len(list_h_ct))
    vertical_extraction_iter_5_img = cv2.morphologyEx(invimg, cv2.MORPH_ERODE, kernel, iterations=3)
    vertical_extraction_buffup_iter_5_img = cv2.morphologyEx(vertical_extraction_iter_5_img, cv2.MORPH_DILATE, kernel,
                                                             iterations=3)
    if major == '3':
        _, ccontours, _ = cv2.findContours(vertical_extraction_buffup_iter_5_img, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    else:
        ccontours, _ = cv2.findContours(vertical_extraction_buffup_iter_5_img, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_NONE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, threshold_dis))
    kernel_erase = vertical_extraction_buffup_iter_5_img + horizontal_extraction_buffup_iter_5_img
    kernel = np.ones((3,3), np.uint8)
    kernel_erase = cv2.dilate(kernel_erase, kernel)
    bit_and_img = cv2.bitwise_and(invimg, kernel_erase)
    if debug == True:
        cv2.imshow('v',vertical_extraction_buffup_iter_5_img)
        cv2.imshow('h',horizontal_extraction_buffup_iter_5_img)
        cv2.imshow("k", kernel_erase)
        cv2.imshow('bitimg', bit_and_img)
    if major == '3':
        _, bcontours, _ = cv2.findContours(bit_and_img, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    else:
        bcontours, _ = cv2.findContours(bit_and_img, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_NONE)
    if debug == True:
        img_dr = img.copy()
        img_dr2 = img.copy()
    list_ct_cluster = []
    list_bb_cluster = []
    for c in bcontours:
        x, y, w, h = cv2.boundingRect(c)
        has_cluster = False
        for l in range(len(list_ct_cluster)):
            wclt = list_ct_cluster[l][0]
            hclt = list_ct_cluster[l][1]
            cnt  = list_ct_cluster[l][2]
            if w/wclt >= 0.9 and w/wclt <= 1.05 and h/hclt >= 0.9 and h/hclt <= 1.05:
                has_cluster = True
                list_ct_cluster[l][0] = (wclt*cnt + w)/(cnt+1)
                list_ct_cluster[l][1] = (hclt*cnt + h)/(cnt+1)
                list_ct_cluster[l][2] += 1
                list_bb_cluster[l].append([x, y, x + w, y + h])
                break
        if has_cluster == False:
            list_ct_cluster.append([w, h, 1])
            list_bb = [[x, y, x + w, y + h]]
            list_bb_cluster.append(list_bb)

    has_square_cell = False
    for i in range(len(list_ct_cluster)):
        if list_ct_cluster[i][2] > 2:
            for bb in list_bb_cluster[i]:
                has_square_cell = True
                cv2.rectangle(blank_image, (bb[0], bb[1]), (bb[2], bb[3]), (255, 255, 255), 3)
                if debug == True:
                    cv2.rectangle(img_dr, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 1)
    if has_square_cell == False:
        blank_image = kernel_erase
    if debug == True:
        cv2.imshow('draw',img_dr)
        cv2.imshow("draw2",img_dr2)
        cv2.imshow("and logic ",blank_image)
    imrs = erase_img(img,blank_image,None,2)
    return imrs


def check_mark(img, scale_box_check=0.7,threshold_cf = 0.08, debug = False):
    # kernel = np.ones((2, 2), np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)
    img_iv = invert_image(img)
    h_im = img_iv.shape[0]
    w_im = img_iv.shape[1]

    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, he = cv2.findContours(img_iv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    else:
        contours, he = cv2.findContours(img_iv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        # print("Uncheckmark")
        return False
    if debug == True:
        img22 = img.copy()
        img_choice = img.copy()
        img_cropbb = img.copy()
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img22,(x,y),(x + w, y + h),(0,0,255),1)
    list_contour_sq = []
    smax = 0
    top_l = None
    btm_r = None
    size_max = None

    smax_sq = 0
    top_lsq = None
    btm_rsq  = None
    size_maxsq = None

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w > 5 or h > 5):
            if float(w) / float(h) < 1.1 and float(w) / float(h) > 0.8:
                if debug == True:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                if w * h > smax:
                    smax = w * h
                    top_l = [x, y]
                    btm_r = [x + w, y + h]
                    size_max = [w, h]
                if y > 0 and y < h_im and x > 0 and x < w_im\
                    and y+h > 0 and y+h < h_im and x + w > 0 and x+w < w_im:
                    if w * h > smax_sq:
                        smax_sq = w * h
                        top_lsq = [x, y]
                        btm_rsq = [x + w, y + h]
                        size_maxsq = [w, h]

    if size_max is None and len(contours) > 0:
        if debug == True:
            cv2.imshow("crop draw", img_cropbb)
            cv2.imshow("iv", img_iv)
            cv2.imshow("img", img)
            cv2.imshow("imgc", img_choice)
            cv2.imshow("img22", img22)
            cv2.waitKey()
        return True
    if size_maxsq is not None:
        top_l = top_lsq
        btm_r = btm_rsq
        size_max = size_maxsq
    if debug == True:
        cv2.rectangle(img_choice, (top_l[0], top_l[1]), (btm_r[0], btm_r[1]), (0, 0, 255), 1)
    w_check = size_max[0] * scale_box_check
    h_check = size_max[1] * scale_box_check

    x_center = top_l[0] + size_max[0] / 2
    y_center = top_l[1] + size_max[1] / 2
    top_l_check = [int(x_center - w_check / 2), int(y_center - h_check / 2)]
    # print(top_l_check)
    btm_r_check = [int(top_l_check[0] + w_check), int(top_l_check[1] + h_check)]
    # print(btm_r_check)
    imcrop_b = crop_image(img_iv, top_l_check[0], top_l_check[1], btm_r_check[0], btm_r_check[1])
    if debug == True:
        cv2.rectangle(img_cropbb,(top_l_check[0],top_l_check[1]), (btm_r_check[0],btm_r_check[1]), (0,0,255),1)
        cv2.imshow("crop draw",img_cropbb)
        cv2.imshow("iv", img_iv)
        cv2.imshow("cr",imcrop_b)
        cv2.imshow("img",img)
        cv2.imshow("imgc",img_choice)
        cv2.imshow("img22",img22)
        cv2.waitKey()
    numb_w = cv2.countNonZero(imcrop_b)
    cv2.rectangle(img, (top_l[0], top_l[1]), (btm_r[0], btm_r[1]), (0, 0, 255), 2)
    cv2.rectangle(img, (top_l_check[0], top_l_check[1]), (btm_r_check[0], btm_r_check[1]), (255, 0, 0), 2)
    # print("W: ", numb_w)
    if numb_w / (w_check * h_check) >= threshold_cf:
        return True
    else:
        return False


def removeLines(img, margin=5, expand_bb=2, b_erase_img=True, pixel_erase=3):
    img_cp = img.copy()
    edges = invert_image(img)
    img_cp2 = img.copy()
    h, w = edges.shape
    lines = cv2.HoughLinesP(edges, 3, np.pi / 180, threshold=90, minLineLength=60, maxLineGap=10)
    list_cluster_lines = []
    max_lenght = 0
    max_width = 0
    bb_line = []
    list_line_max = []
    # struct cluster y average, numb point incluster, x top right ,y top right, x bottom left, y bottom left
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        cv2.line(img_cp, (x1, y1), (x2, y2), (0, 0, 255), 2)
        ytb = int((y1 + y2) / 2)
        hascluster = False
        for cluster in list_cluster_lines:
            if abs(ytb - cluster[0]) < margin:
                cluster[0] = (cluster[0] * cluster[1] + ytb) / (cluster[1] + 1)
                cluster[1] += 1
                cluster[2] = min(cluster[2], x1, x2)
                cluster[3] = min(cluster[3], y1, y2)
                cluster[4] = max(cluster[4], x1, x2)
                cluster[5] = max(cluster[5], y1, y2)
                cluster[6].append([x1, y1, x2, y2])
                if (cluster[4] - cluster[2]) > max_lenght and (cluster[5] - cluster[3]) >= max_width:
                    max_lenght = (cluster[4] - cluster[2])
                    max_width = (cluster[5] - cluster[3])
                    bb_line = [cluster[2], cluster[3] - expand_bb, cluster[4], cluster[5] + expand_bb]
                    list_line_max = cluster[6]
                hascluster = True
                break
        if hascluster == False:
            new_cluster = [ytb, 1, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            list_line = []
            list_line.append([x1, y1, x2, y2])
            new_cluster.append(list_line)
            list_cluster_lines.append(new_cluster)
            if (new_cluster[4] - new_cluster[2]) >= max_lenght and (new_cluster[5] - new_cluster[3]) >= max_width:
                max_lenght = (new_cluster[4] - new_cluster[2])
                max_width = (new_cluster[5] - new_cluster[3])
                list_line_max = new_cluster[6]
                bb_line = [new_cluster[2], new_cluster[3] - expand_bb, new_cluster[4], new_cluster[5] + expand_bb]
    if (bb_line[2] - bb_line[0]) / img.shape[2] > 0.7:
        if b_erase_img == True:
            blank_image = np.zeros(shape=[img_cp2.shape[0], img_cp2.shape[1], 1], dtype=np.uint8)
            img_test = img.copy()
            for ln in list_line_max:
                cv2.line(img_test, (ln[0], ln[1]), (ln[2], ln[3]), (0, 0, 255), expand_bb)
                cv2.line(blank_image, (ln[0], ln[1]), (ln[2], ln[3]), (255, 255, 255), expand_bb)
            img_cp2 = erase_img(img_cp2, blank_image, bb_line, pixel_erase, cv2.INPAINT_NS)
        else:
            crop_img = crop_image(img_cp2, 0, 0, bb_line[2] - bb_line[0], bb_line[3] - bb_line[1])
            img_cp2[bb_line[1]:bb_line[3], bb_line[0]:bb_line[2]] = crop_img
        return img_cp2
    else:
        return None



def crop_fit_text(img, expand=5):
    img2 = img.copy()
    h_img = img.shape[0]
    w_img = img.shape[1]
    image_i = invert_image(img)
    lines = cv2.HoughLinesP(image_i, 3, np.pi / 180, threshold=90, minLineLength=10, maxLineGap=3)
    blank_image = np.zeros(shape=[image_i.shape[0], image_i.shape[1], 1], dtype=np.uint8)
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        cv2.line(blank_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 1))
    h_img_b = cv2.morphologyEx(blank_image, cv2.MORPH_ERODE, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 3))
    h_img_b = cv2.morphologyEx(h_img_b, cv2.MORPH_DILATE, kernel,
                               iterations=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 70))
    v_img = cv2.morphologyEx(blank_image, cv2.MORPH_ERODE, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 70))
    v_img = cv2.morphologyEx(v_img, cv2.MORPH_DILATE, kernel,
                             iterations=3)
    blank_image = h_img_b + v_img
    img_erase = erase_img(img, blank_image, 5)

    img_gray = cv2.cvtColor(img_erase, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    xtop_left = 10e9
    ytop_left = 10e9
    xbt_right = -1
    ybt_right = -1
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 1)
        if w / h < 3 and h / w < 3 and w > 10 and h > 10:
            if x < xtop_left:
                xtop_left = x
            if y < ytop_left:
                ytop_left = y
            if x + w > xbt_right:
                xbt_right = x + w
            if y + h > ybt_right:
                ybt_right = y + h
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 1)
    img_crop = crop_image(img_erase, max(0, xtop_left - expand), max(0, ytop_left - expand), \
                          min(xbt_right + expand, w_img), min(ybt_right + expand, h_img))
    return img_crop


def list_files1(directory, extension):
    # print(listdir(directory))
    list_file = []
    for f in listdir(directory):
        if f.endswith('.' + extension):
            list_file.append(f)
    return list_file



def test_jpg():
    img_raw=cv2.imread('/home/aicr/cuongnd/aicr.core/test/template_4.png')
    cv2.imwrite('/home/aicr/cuongnd/aicr.core/test/template_4.jpg',img_raw)

def gen_mask_img(path):
    training_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
    path = path
    save_path = path + '_mark'
    print(save_path)
    if save_path is not None:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
    list_file = list_files1(path, 'jpg')
    list_file += list_files1(path, 'png')
    for pimg in list_file:
        path_img = os.path.join(path, pimg)
        path_img_s = os.path.join(save_path, pimg)
        path_txt = path_img.split('.')[0] + '.txt'
        list_point_contour = []
        with open(path_txt, mode='r+', encoding='utf-8') as readf:
            count = 0
            for line in readf:
                list_str = line.split(',')
                xtl = int(list_str[0])
                ytl = int(list_str[1])
                xtr = int(list_str[2])
                ytr = int(list_str[3])
                xbr = int(list_str[4])
                ybr = int(list_str[5])
                xbl = int(list_str[6])
                ybl = int(list_str[7])
                penta = np.array([[xtl, ytl], [xtr, ytr], [xbr, ybr], [xbl, ybl]], np.int32)
                list_point_contour.append(penta)
        img = cv2.imread(path_img)
        h = img.shape[0]
        w = img.shape[1]
        blank_image = np.zeros(shape=[h, w, 1], dtype=np.uint8)
        for c in list_point_contour:
            cv2.fillPoly(blank_image, [c], (255, 255, 255))
        print(path_img_s)
        cv2.imwrite(path_img_s, blank_image)
        path_img_new = path_img.split('.')[0] + '_' + training_time + '.' + path_img.split('.')[1]
        path_img_s_new = path_img_s.split('.')[0] + '_' + training_time + '.' + path_img_s.split('.')[1]
        print(path_img_s_new)
        cv2.imwrite(path_img_s_new, blank_image)
        os.rename(path_img, path_img_new)

def resize_padding(img,size):
    h = img.shape[0]
    w = img.shape[1]
    scale = size/w
    blank_image = np.zeros(shape=[size, size, 1], dtype=np.uint8)
    blank_image = 255 - blank_image
    blank_image = cv2.cvtColor(blank_image,cv2.COLOR_GRAY2BGR)
    if h > w:
        scale = size/h
    print(blank_image.shape)
    imgrs = cv2.resize(img,None,fx = scale, fy = scale,interpolation=cv2.INTER_CUBIC)
    offsetx = int(( size - imgrs.shape[1])/2.)
    offsety = int(( size - imgrs.shape[0])/2.)
    print(offsetx,offsety)
    print(imgrs.shape)
    blank_image[ offsety : offsety + imgrs.shape[0] , offsetx : offsetx + imgrs.shape[1] ] = imgrs
    return blank_image


if __name__ == "__main__":
    import glob
    files = [f for f in glob.glob('C:/Users/chungnx/Desktop/test_face/*.jpg', recursive=True)]
    for l in files:
        img = cv2.imread(l)
        img = resize_padding(img,224)
        cv2.imwrite(l,img)
    # convertTXT2JSON('C:/Users/chungnx/Desktop/relable/002_template.txt','C:/Users/chungnx/Desktop/relable/002_template.json')
    # path = 'C:/Users/chungnx/Desktop/test'
    # list_file = list_files1(path,'jpg')
    # list_file += list_files1(path,'png')
    # for pimg in list_file:
    #     path_img = os.path.join(path,pimg)
    #     img = cv2.imread(path_img)
    #     img_r = auto_rotation(img,2)
    #     path_img = path_img.replace('.','r.')
    #     cv2.imwrite(path_img, img_r)

    # img_test = cv2.resize(img_test, (2480, 3508))
    # img_calib = calib_image(img_test)
    # # cv2.imwrite("img_cal.jpg",img_calib)
    # f = open("C:/Users/chungnx/PycharmProjects/aicr.core/classifier_CRNN/form/template.json", mode='r', encoding='utf-8')
    # json_content = f.read()
    # list_clImageInfor_demo, max_wh,lis_mark = extract_for_demo_json(img_calib,json_content , eraseline=False,\
    #                                                   subtract_bgr=False, gen_bgr=False)
    # list_img = list_files1(path, "jpg")
    # list_img += list_files1(path, "png")
    # img_bg = cv2.imread(path_bg)
    # for l in list_img:
    #     path_img = os.path.join(path, l)
    #     img = cv2.imread(path_img)
    #     check_mark(img)
    #     cv2.waitKey()

