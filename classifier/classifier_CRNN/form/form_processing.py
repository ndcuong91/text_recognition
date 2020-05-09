import cv2
import numpy as np


def visualize_boxes(path_config_file, img, debug=False, offset_x=0, offset_y=0):
    try:
        with open(path_config_file, 'r+') as readf:
            count = 0
            for line in readf:
                count += 1
                list_inf = line.split()
                if len(list_inf) == 5:
                    last_idx = len(list_inf) - 1
                    bb = [int(list_inf[last_idx - 3]), int(list_inf[last_idx - 2]),
                          int(list_inf[last_idx - 3]) + int(list_inf[last_idx - 1]),
                          int(list_inf[last_idx - 2]) + int(list_inf[last_idx])]
                    wh_ratio = float(list_inf[last_idx - 1]) / float(list_inf[last_idx])
                    # print(count, list_inf[0], 'wh_ratio', wh_ratio)
                    # cv2.putText(img, list_inf[0], (bb[0], bb[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2,
                    #            cv2.LINE_AA)
                    cv2.rectangle(img, (bb[0] + offset_x, bb[1] + offset_y), (bb[2]+ offset_x, bb[3]+ offset_y), (0, 0, 255), 2)
    except:
        pass

    img_res = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
    if debug:
        cv2.imshow('result', img_res)
        cv2.waitKey(0)
    return img_res


def visualize_boxes_json(temp_json, img, debug=False):
    try:
        data_dict = temp_json
        dict_fields = data_dict['fields']
        h_n = img.shape[0]
        w_n = img.shape[1]
        ratioy = h_n / data_dict['image_size']['height']
        ratiox = w_n / data_dict['image_size']['width']
        for df in dict_fields:
            if df['type'] != 'mark':
                by = int(df['position']['top'])
                bx = int(df['position']['left'])
                ex = bx + int(df['size']['width'])
                ey = by + int(df['size']['height'])

                bx = int(bx * ratiox)
                ex = int(ex * ratiox)
                by = int(by * ratioy)
                ey = int(ey * ratioy)
                # print(df['label'], bx, by, ex-bx, ey-by)

                cv2.rectangle(img, (bx, by), (ex, ey), (0, 0, 255), 2)
    except:
        pass

    img_res = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
    if debug:
        cv2.imshow('result', img_res)
        cv2.waitKey(0)
    return img_res


import os
def erose(root_dir, img_list):
    for img_path in img_list:
        img_path = os.path.join(root_dir, img_path)
        img = cv2.imread(img_path, 0)
        kernel = np.ones((1, 5), np.uint8)
        erosion = cv2.erode(img, kernel, iterations=1)
        cv2.imshow('erode', erosion)
        ch = cv2.waitKey(0)
        if ch == 27:
            print('Saved', img_path)
            cv2.imwrite(img_path, erosion)


if __name__ == "__main__":
    img = cv2.imread('template_VIB/0001_ori.jpg')
    # visualize_boxes('template_VIB_page1.txt', img, debug=True)
    root_dir = 'background_VIB'
    img_list = ['6.jpg', '11.jpg', '12.jpg', '38.jpg']
    erose(root_dir, img_list)
