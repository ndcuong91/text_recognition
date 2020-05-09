import cv2
import numpy as np
import torch
from PIL import Image
import pickle
import math
import os
from datetime import datetime
import random
import shutil
from torchvision import transforms

try:
    from pre_processing.image_preprocessing import erase_img, crop_image, rotate_image_angle, invert_image, list_files1
    from pre_processing.table_border_extraction_fns import get_h_and_v_line_bbox, \
        clTable, get_h_and_v_line_bbox_CNX, filter_lines_error, detect_table
except ImportError:
    from image_preprocessing import erase_img, crop_image, rotate_image_angle, invert_image, list_files1
    from table_border_extraction_fns import get_h_and_v_line_bbox, \
        clTable, get_h_and_v_line_bbox_CNX, filter_lines_error, detect_table


def augment_image_addline(img, color=(60, 60, 60), size_draw=1, range_dot=15, sizelinedot=1, random_pos=False):
    type = random.randint(-1, 2)
    white_color = (70, 127, 209)
    # print('augment_image_addline. Type', type)
    # type 0 solid line, 1 dots line,2 line dot
    img2 = img.copy()
    h_img = img.shape[0]
    w_img = img.shape[1]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    sum_h = 0
    count = 0
    listbb = []
    if random_pos == False:
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if (w > 20 and h > 20):
                sum_h += (y + h)
                count += 1
                listbb.append([x, y, x + w, y + h])
    if count > 0 and random_pos == False:
        h_average = int(sum_h / count)
    else:
        h_average = random.randint(int(h_img / 5), int(4 * h_img / 5))

    if type <= 0:
        black = random.choice([True, False])
        if black:
            # print('black')
            cv2.line(img2, (0, h_average - 1), (w_img, h_average - 1), color, int(size_draw / 3), cv2.LINE_AA)
            cv2.line(img2, (0, h_average), (w_img, h_average), color, int(size_draw / 3), cv2.LINE_AA)
            cv2.line(img2, (0, h_average + 1), (w_img, h_average + 1), color, int(size_draw / 3), cv2.LINE_AA)
        else:
            # print('white')
            cv2.line(img2, (0, h_average - 2), (w_img, h_average - 2), white_color, int(size_draw / 3), cv2.LINE_AA)
            cv2.line(img2, (0, h_average - 1), (w_img, h_average - 1), white_color, int(size_draw / 3), cv2.LINE_AA)
            cv2.line(img2, (0, h_average), (w_img, h_average), white_color, int(size_draw / 3), cv2.LINE_AA)

    elif type == 1 or type == 2:
        for i in range(int(w_img / range_dot)):
            if type == 1:
                centerx = int((i * range_dot) + int(size_draw / 2))
                centery = h_average - int((size_draw) / 2)
                cv2.circle(img2, (centerx, centery), size_draw, color, -1)
            elif type == 2:
                size_line = int(range_dot / sizelinedot)
                beginl = int(i * range_dot) + int(i * size_line)
                endl = beginl + size_line
                cv2.line(img2, (beginl, h_average - 1), (endl, h_average - 1), color, int(size_draw / 3), cv2.LINE_AA)
                cv2.line(img2, (beginl, h_average), (endl, h_average), color, int(size_draw / 3), cv2.LINE_AA)
                cv2.line(img2, (beginl, h_average + 1), (endl, h_average + 1), color, int(size_draw / 3), cv2.LINE_AA)
    return img2


def augment_random_rotate(img, begin, end, pixel_erase=3):
    angle = random.randrange(begin, end)
    img2 = rotate_image_angle(img, angle)
    return img2


def augment_to_sepia(img):
    sepiakn = np.matrix([[0.272, 0.534, 0.131],
                         [0.272, 0.534, 0.131],
                         [0.393, 0.769, 0.189]])
    imgrs = cv2.transform(img, sepiakn)
    return imgrs


def augment_add_hsv(img, light=30, sat=26):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    if light < 0:
        img_hsv[:, :, 2] -= abs(light)
    elif light > 0:
        img_hsv[:, :, 2] += abs(light)

    if sat < 0:
        img_hsv[:, :, 1] -= abs(sat)
    elif sat > 0:
        img_hsv[:, :, 1] += abs(sat)
    img_hsv = np.clip(img_hsv, 0, 255)
    imgbgr = cv2.cvtColor(img_hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return imgbgr


def augment_random_erase(img, numb_char_erase):
    img2 = img.copy()
    h_img = img.shape[0]
    w_img = img.shape[1]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    sum_h = 0
    count = 0
    listbb = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w > 20 and h > 20):
            sum_h += (y + h)
            count += 1
            listbb.append([x, y, x + w, y + h])
    id_box_erase = random.sample(range(0, len(listbb) - 1), numb_char_erase)
    print(id_box_erase)
    for id in id_box_erase:
        xb, yb, xe, ye = listbb[id]
        xb = random.randrange(xb, int((xe + xb) / 2))
        xe = random.randrange(xb + 1, xe)
        yb = random.randrange(yb, ye)
        img2 = erase_img(img2, None, [xb, yb, xe, ye], 10)
    return img2


def augment_resize_img(img, w, h, fsx=None, fsy=None):
    if fsx is None and fsy is None:
        imgrs = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return imgrs
    else:
        imgrs = cv2.resize(img, None, fx=fsx, fy=fsy, interpolation=cv2.INTER_AREA)
        return imgrs


def augment_bold_characters(img, kernel_dms=3, iter=1):
    # print('augment_bold_characters')
    kernel = np.ones((kernel_dms, kernel_dms), np.uint8)
    img_bold = cv2.erode(img, kernel, iterations=iter)
    return img_bold


def augment_thin_characters(img, kernel_dms=3, iter=1):
    # print('augment_thin_characters')
    kernel = np.ones((kernel_dms, kernel_dms), np.uint8)
    img_thin = cv2.dilate(img, kernel, iterations=iter)
    return img_thin


def augment_blur(img, type_blur=0, size_window=5):
    # type 0 average bluer, 1 gaussian blur, 2 median blur
    img_rs = None
    if type_blur == 0:
        img_rs = cv2.blur(img, (size_window, size_window))
    elif type_blur == 1:
        img_rs = cv2.GaussianBlur(img, (size_window, size_window), 0)
    else:
        img_rs = cv2.medianBlur(img, size_window)
    return img_rs


def augment_add_noise(image, prob=0.02):
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(image.shape[:2])
    image[probs < (prob / 2)] = black
    image[probs > 1 - (prob / 2)] = white
    return image


def aument_gen_list_box_number(img, img_bg=None, expand_size=10, expand_size_char=3, size_line_board=2,
                               color_text=None, ):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    img_edges = invert_image(img)
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    list_bb_ct = []

    img_draw = img.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 5:
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 0, 255), 1)
            list_bb_ct.append([x, y, x + w, y + h, w, h])
    # clean bb
    len_list = len(list_bb_ct)
    check_list = [0] * len_list
    for i in range(len_list):
        if check_list[i] == 1:
            continue
        bx1, by1, ex1, ey1, _, _ = list_bb_ct[i]
        for j in range(i + 1, len_list):
            if check_list[j] == 1:
                continue
            bx2, by2, ex2, ey2, _, _ = list_bb_ct[j]
            if bx1 >= bx2 and by1 >= by2 \
                    and ex1 <= ex2 and ey1 <= ey2:
                check_list[i] = 1
                break
            elif bx2 >= bx1 and by2 >= by1 \
                    and ex2 <= ex1 and ey2 <= ey1:
                check_list[j] = 1

    list_bb_final = []
    wmax = 0
    hmax = 0
    for i in range(len_list):
        if check_list[i] == 0:
            list_bb_final.append(list_bb_ct[i])
            if list_bb_ct[i][4] > wmax:
                wmax = list_bb_ct[i][4]
            if list_bb_ct[i][5] > hmax:
                hmax = list_bb_ct[i][5]
    list_bb_final.sort(key=lambda x: x[0])
    len_final = len(list_bb_final)
    max_size = max(hmax, wmax) + expand_size_char
    w_img = len_final * max_size + (len_final + 1) * size_line_board + expand_size * 2
    h_img = max_size + 2 * size_line_board + expand_size * 2
    background_img = None
    if img_bg is not None:
        background_img = cv2.resize(img_bg, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
    else:
        background_img = np.zeros(shape=[h_img, w_img, 3], dtype=np.uint8)
        background_img = np.full_like(background_img, fill_value=255)
    begin_point = None
    yb = int(expand_size + size_line_board / 2)
    ye = yb + max_size
    for i in range(len_final):
        xb = int(expand_size + size_line_board / 2 + max_size * i)
        xe = xb + max_size
        cv2.rectangle(background_img, (xb, yb), (xe, ye), (10, 23, 13), int(size_line_board))
        xb_insert_img = int(xb + max_size / 2 - list_bb_final[i][4] / 2)
        yb_insert_img = int(yb + max_size / 2 - list_bb_final[i][5] / 2)
        crp_img_raw = crop_image(img, list_bb_final[i][0], list_bb_final[i][1], list_bb_final[i][2],
                                 list_bb_final[i][3])
        crp_img_bi = invert_image(crp_img_raw)
        h_img_cr = crp_img_raw.shape[0]
        v_img_cr = crp_img_raw.shape[1]
        if color_text is None:
            for i in range(h_img_cr):
                for j in range(v_img_cr):
                    if crp_img_bi[i, j] > 0:
                        background_img[yb_insert_img + i, xb_insert_img + j] = crp_img_raw[i, j]
        else:
            for i in range(h_img_cr):
                for j in range(v_img_cr):
                    if crp_img_bi[i, j] > 0:
                        background_img[yb_insert_img + i, xb_insert_img + j] = color_text
    return background_img


def change_background_handwriting(img_bg, img, expand=5, color_text=None):
    img2 = img.copy()
    h_img = img.shape[0]
    w_img = img.shape[1]
    img_bg_rs = cv2.resize(img_bg, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
    print(img_bg_rs.shape)
    print(img2.shape)

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
    img_erase = erase_img(img, blank_image, expand)
    img_v = invert_image(img_erase)
    if color_text == None:
        for i in range(h_img):
            for j in range(w_img):
                if img_v[i, j] > 0:
                    img_bg_rs[i, j] = img2[i, j]
    else:
        for i in range(h_img):
            for j in range(w_img):
                if img_v[i, j] > 0:
                    img_bg_rs[i, j] = color_text
    return img_bg_rs


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


def augment_randomResizePadding(img, width, height, min_scale=1 / 2, max_scale=2, fill=(255, 255, 255), train=True):
    desired_w, desired_h = width, height  # (width, height)
    img_w, img_h = img.size  # old_size[0] is in (width, height) format
    stretch_scale = 1.0
    if train:
        stretch_scale = random.uniform(min_scale, max_scale)
    # print('augment_randomResizePadding. Scale:', stretch_scale)
    ratio = stretch_scale * img_w / img_h
    new_w = int(desired_h * ratio)
    new_w = new_w if desired_w == None else min(desired_w, new_w)
    img = img.resize((new_w, desired_h), Image.ANTIALIAS)

    # padding image
    # offsetx = 0
    # if train:
    #     offsetx = max(0, int((desired_w - new_w) / 2) -2)
    # if desired_w != None and desired_w > new_w:
    #     new_img = Image.new("RGB", (desired_w, desired_h), color=fill)
    #     new_img.paste(img, (offsetx, 0))
    #     img = new_img

    if desired_w != None and desired_w > new_w:
        new_img = Image.new("RGB", (desired_w, desired_h), color=fill)
        new_img.paste(img, (0, 0))
        img = new_img
    print(img.size)
    return img


def augment_resizePadding(img, width, height, fill=(255, 255, 255), train=True):
    desired_w, desired_h = width, height  # (width, height)
    img_w, img_h = img.size  # old_size[0] is in (width, height) format
    stretch_scale = 1.0
    if train:
        stretch_scale = random.uniform(2 / 3, 2)
    ratio = stretch_scale * img_w / img_h
    new_w = int(desired_h * ratio)
    new_w = new_w if desired_w == None else min(desired_w, new_w)
    img = img.resize((new_w, desired_h), Image.ANTIALIAS)

    # padding image
    if desired_w != None and desired_w > new_w:
        new_img = Image.new("RGB", (desired_w, desired_h), color=fill)
        new_img.paste(img, (0, 0))
        img = new_img

    return img


def augment_image_addline_simple(img, size_draw=1, sizelinedot=1):
    type = random.randint(-1, 2)
    black_color = (10, 10, 10)
    red_color = (70, 127, 209)
    white_color = (255, 255, 255)
    rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    range_dot = random.randint(5,15)
    #print('augment_image_addline_simple. Type', type)
    # type <0 solid line, 1 dots line,2 line dot
    h_img = img.shape[0]
    w_img = img.shape[1]

    h_average = random.randint(int(h_img / 5), int(4 * h_img / 5))
    size_draw = max(int(size_draw / 3), 1)
    #print('size draw',size_draw)

    if type <= 0:
        draw_color = random.choice([black_color, red_color, red_color, white_color, rand_color]) #double red color
        #print(draw_color)
        cv2.line(img, (0, h_average - 1), (w_img, h_average - 1), draw_color, size_draw)
        cv2.line(img, (0, h_average), (w_img, h_average), draw_color, size_draw)
        cv2.line(img, (0, h_average + 1), (w_img, h_average + 1), draw_color, size_draw)
    else:
        draw_color = random.choice([black_color, red_color, rand_color])
        #print(draw_color)
        for i in range(int(w_img / range_dot)):
            if type == 1:
                centerx = int((i * range_dot) + int(size_draw / 2))
                centery = h_average - int((size_draw) / 2)
                cv2.circle(img, (centerx, centery), size_draw, draw_color, -1,cv2.LINE_AA)
            elif type == 2:
                size_line = int(range_dot / sizelinedot)
                beginl = int(i * range_dot) + int(i * size_line)
                endl = beginl + size_line
                cv2.line(img, (beginl, h_average - 1), (endl, h_average - 1), draw_color, size_draw)
                cv2.line(img, (beginl, h_average), (endl, h_average), draw_color, size_draw)
                cv2.line(img, (beginl, h_average + 1), (endl, h_average + 1), draw_color, size_draw)
    return img


######################################## gen data script ##################################

def gen_data_number_with_box(path, path_save):
    list_img = ["3.jpg", "4.jpg", "5.jpg", "6.jpg", "10.jpg", "11.jpg", "12.jpg", "15.jpg",
                "16.jpg", "17.jpg", "18.jpg", "21.jpg", "22.jpg", "23.jpg", "24.jpg"]
    list_txt = ["3.txt", "4.txt", "5.txt", "6.txt", "10.txt", "11.txt", "12.txt", "15.txt",
                "16.txt", "17.txt", "18.txt", "21.txt", "22.txt", "23.txt", "24.txt"]
    list_path = os.listdir(path)
    for pathd in list_path:
        path_data = os.path.join(path, pathd)
        path_save_data = os.path.join(path_save, pathd)
        if not os.path.isdir(path_save_data):
            os.mkdir(path_save_data)
        for i in range(len(list_img)):
            gt_path = os.path.join(path_data, list_txt[i])
            img_pth = os.path.join(path_data, list_img[i])
            shutil.copy(gt_path, path_save_data)
            img = cv2.imread(img_pth)
            if img.shape[0] == 0 or img.shape[1] == 0:
                continue
            img = crop_fit_text(img)
            if img.shape[0] == 0 or img.shape[1] == 0:
                continue
            imgrs = aument_gen_list_box_number(img, expand_size_char=1)
            path_save_img = os.path.join(path_save_data, list_img[i])
            cv2.imwrite(path_save_img, imgrs)


def gen_data_path(path, path_save):
    list_img = list_files1(path, "jpg")
    list_img += list_files1(path, "png")
    dir_name = os.path.dirname(path)
    save_dir_line = os.path.join(path_save, dir_name + "_lines")

    if not os.path.isdir(save_dir_line):
        os.mkdir(save_dir_line)

    save_dir_dot = os.path.join(path_save, dir_name + "_dots")
    if not os.path.isdir(save_dir_dot):
        os.mkdir(save_dir_dot)

    save_dir_ldot = os.path.join(path_save, dir_name + "_linedots")
    if not os.path.isdir(save_dir_ldot):
        os.mkdir(save_dir_ldot)

    for img_path in list_img:
        path_img = os.path.join(path, img_path)
        print(path_img)
        img = cv2.imread(path_img)
        # line
        rs = augment_image_addline(img, type=0)
        path_save = os.path.join(save_dir_line, img_path)
        cv2.imwrite(path_save, rs)
        # line
        rs = augment_image_addline(img, type=1)
        path_save = os.path.join(save_dir_dot, img_path)
        cv2.imwrite(path_save, rs)
        # line
        rs = augment_image_addline(img, type=2)
        path_save = os.path.join(save_dir_ldot, img_path)
        cv2.imwrite(path_save, rs)


def gen_white_data(path, path_save):
    list_img = list_files1(path, "jpg")
    list_img += list_files1(path, "png")
    dir_name = os.path.basename(path)
    save_dir_white_line = os.path.join(path_save, 'add_solid_white_line', dir_name)

    if not os.path.isdir(save_dir_white_line):
        os.mkdir(save_dir_white_line)
    count = 0
    for img_name in list_img:
        path_img = os.path.join(path, img_name)
        count += 1
        print(count, path_img)
        img = cv2.imread(path_img)

        anno_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        anno_path_src = os.path.join(path, anno_name)
        # solid
        rs = augment_image_addline(img, 0, black=False)
        img_path_save = os.path.join(save_dir_white_line, img_name)
        anno_path_save = os.path.join(save_dir_white_line, anno_name)
        cv2.imwrite(img_path_save, rs)
        shutil.copy(anno_path_src, anno_path_save)


def gen_random_rotate(path, path_save):
    list_img = list_files1(path, "jpg")
    list_img += list_files1(path, "png")
    dir_name = os.path.basename(path)
    save_dir_rotate = os.path.join(path_save, 'add_rotate', dir_name)

    if not os.path.isdir(save_dir_rotate):
        os.mkdir(save_dir_rotate)

    count = 0
    for img_name in list_img:
        path_img = os.path.join(path, img_name)
        count += 1
        print(count, path_img)
        img = cv2.imread(path_img)

        anno_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        anno_path_src = os.path.join(path, anno_name)
        # solid
        rs = augment_random_rotate(img, -5, 5)
        img_path_save = os.path.join(save_dir_rotate, img_name)
        anno_path_save = os.path.join(save_dir_rotate, anno_name)
        cv2.imwrite(img_path_save, rs)
        shutil.copy(anno_path_src, anno_path_save)


####################################### Class for pytorch #######################################
class cnx_aug_add_line(object):
    def __init__(self, color=(60, 60, 60), size_draw=3, range_dot=15, sizelinedot=1):
        # type 0 solid line, 1 dots line,2 line dot
        self.size_draw = size_draw
        self.range_dot = range_dot
        self.sizelinedot = sizelinedot
        self.color = color

    def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_image_addline(cv_img, self.color, self.size_draw, self.range_dot, self.sizelinedot)
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)

class cnd_aug_add_line(object):
    def __init__(self, size_draw=1, sizelinedot=3):
        # type 0 solid line, 1 dots line,2 line dot
        self.size_draw = size_draw
        self.sizelinedot = sizelinedot

    def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_image_addline_simple(cv_img, self.size_draw, self.sizelinedot)
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)


class cnx_aug_autorotate(object):
    def __init__(self, begin_ag, end_ag, pixel_erase=3):
        self.angle_bg = begin_ag
        self.angle_end = end_ag
        self.pixel_erase = pixel_erase

    def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_random_rotate(cv_img, self.angle_bg, self.angle_end, self.pixel_erase)
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)


class cnx_aug_random_erase(object):
    def __init__(self, numb_char_erase):
        self.numb_char_erase = numb_char_erase

    def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_random_erase(cv_img, self.numb_char_erase)
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)


class cnx_aug_bold_characters(object):
    def __init__(self, kernel_dms=3, iter=1):
        self.kernel_dms = kernel_dms
        self.iter = iter

    def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_bold_characters(cv_img, self.kernel_dms, self.iter)
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)


class cnx_aug_thin_characters(object):
    def __init__(self, kernel_dms=3, iter=1):
        self.kernel_dms = kernel_dms
        self.iter = iter

    def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_thin_characters(cv_img, self.kernel_dms, self.iter)
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)


class cnx_aug_sepia_effect(object):
    def __init__(self):
        pass

    def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_to_sepia(cv_img)
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)


class cnx_aug_add_hsv(object):
    def __init__(self, light=30, sat=26):
        self.light = light
        self.sat = sat

    def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_add_hsv(cv_img, self.light, self.sat)
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)


class cnx_aug_blur(object):
    def __init__(self, type_blur=0):
        # type 0 average bluer, 1 gaussian blur, 2 median blur
        self.type_blur = type_blur

    def __call__(self, img):
        numpy_image = np.array(img)
        size_window = random.choices([3,5])
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_blur(cv_img, self.type_blur, size_window[0])
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)


class cnx_aug_add_noise(object):
    def __init__(self, prob=0.02):
        self.prob = prob
        pass

    def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_add_noise(cv_img, self.prob)
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)


class cnx_aug_resize_img(object):
    def __init__(self, w, h, fsx=None, fsy=None):
        self.w = w
        self.h = h
        self.fsx = fsx
        self.fsy = fsy

    def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_resize_img(cv_img, self.w, self.h, self.fsx, self.fsy)
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)


class cnd_aug_randomResizePadding(object):
    def __init__(self, width, height, min_scale=1 / 2, max_scale=2, fill=(255, 255, 255), train=True):
        self.width = width
        self.height = height
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.fill = fill
        self.train = train

    def __call__(self, img):  # img is PIL image
        img_rs = augment_randomResizePadding(img, self.width, self.height, self.min_scale, self.max_scale, self.fill,
                                             self.train)
        return img_rs


class cnd_aug_resizePadding(object):
    def __init__(self, width, height, fill=(255, 255, 255), train=True):
        self.width = width
        self.height = height
        self.fill = fill
        self.train = train

    def __call__(self, img):  # img is PIL image
        img_rs = augment_resizePadding(img, self.width, self.height, self.fill, self.train)
        return img_rs


def test_augment():
    from torchvision.transforms import RandomApply, ColorJitter, RandomAffine, ToTensor, Normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    fill_color = (255, 255, 255)  # (209, 200, 193)
    min_scale, max_scale = 2 / 3, 2
    imgW = 1024
    imgH = 64
    transform_train = transforms.Compose([
        # RandomApply([cnx_aug_thin_characters()], p=0.2),
        # RandomApply([cnx_aug_bold_characters()], p=0.4),
        # cnd_aug_randomResizePadding(imgH, imgW, min_scale, max_scale, fill=fill_color),
        cnd_aug_resizePadding(imgW, imgH, fill=fill_color),
        RandomApply([cnd_aug_add_line()], p=0.5),
        RandomApply([cnx_aug_blur()], p=0.3),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        RandomApply([RandomAffine(shear=(-20, 20),
                                  translate=(0.0, 0.1),
                                  degrees=0,
                                  # degrees=2,
                                  # scale=(0.8, 1),
                                  fillcolor=fill_color)], p=0.5)
        # ,ToTensor()
        # ,Normalize(mean, std)
    ])
    path = "/data/train_data_29k_29Feb_update30Mar/cinnamon_data/cinamon_test_115/"
    # path='/data/SDV/cropped_img'
    list_img = list_files1(path, "jpg")
    print(list_img)
    for l in list_img:
        path_img = os.path.join(path, l)
        img = cv2.imread(path_img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_rs = transform_train(img)
        numpy_image = np.array(img_rs)
        # numpy_image = augment_random_rotate(img, -5, 5)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", cv_img)
        cv2.waitKey(0)



if __name__ == "__main__":
    # test_augment()
    img = cv2.imread('C:/Users/chungnx/PycharmProjects/aicr.core/data/SDV_invoices_mod/006_1.jpg')
    cv2.imshow("s",img)
    img1 = filter_color_image(img,lower_red1,upper_red1)
    img2 = filter_color_image(img,lower_red2,upper_red2)
    img3 =  filter_color_image(img,lower_black,upper_black)
    img1 += img2
    img1 = cv2.bitwise_not(img1,mask=img3)
    cv2.imwrite('2.jpg',img1)
    # path = 'C:/Users/chungnx/PycharmProjects/aicr.core/data/cmnd_1/22.jpg'
    # img = cv2.imread(path)
    # cv2.imshow("eq1", img)
    # imgrs = augment_add_noise(img)
    # cv2.imshow("eq4", imgrs)
    # lower_red = np.array([0, 0, 0])
    # upper_red = np.array([255, 255, 120])
    # rs = filter_color_image(img,lower_red,upper_red)
    # cv2.imshow("eq", rs)
    # cv2.waitKey()
