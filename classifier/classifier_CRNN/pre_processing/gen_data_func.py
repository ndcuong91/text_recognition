import json
import cv2
import classifier_CRNN.pre_processing.image_preprocessing as impr
import os
import datetime
import classifier_CRNN.pre_processing.table_border_extraction_fns as tbdetect

def gen_image_for_demo(path_image,
                       temp_json,
                       save_path='/data/data_imageVIB/1/',
                       save_filename='result.jpg',
                       show_text=False):
    if save_path is not None:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
    data_dict = temp_json
    template = impr.clTemplate_demo()
    template.height = data_dict['image_size']['height']
    template.width = data_dict['image_size']['width']
    template.name_template = data_dict['name']
    template.type = data_dict['type']
    template.category = data_dict['category']
    template.imageSource = data_dict['image_source']
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_bl = cv2.imread(path_image)
    img_bl = cv2.resize(img_bl, (template.width, template.height))
    #img_bl = calib_image(img_bl)
    dict_fields = data_dict['fields']
    h_n = img_bl.shape[0]
    w_n = img_bl.shape[1]
    ratioy = h_n / template.height
    ratiox = w_n / template.width
    img_save_bl = img_bl.copy()
    max_wh_ratio = 1
    for df in dict_fields:
        if df['type'] != 'mark':
            classimginf = impr.clImageInfor_demo()
            classimginf.id = df['id']
            classimginf.name = df['name']
            classimginf.label = df['label']
            classimginf.type = df['type']
            classimginf.data_type = df['data_type'] if 'data_type' in df else 'text'
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
            classimginf.location = [bx, by, ex, ey]
            offsetx, offsety = 1, 1
            cv2.rectangle(img_save_bl, (bx, by), (ex, ey), (0, 0, 255), 4)
            if show_text:
                cv2.putText(img_save_bl, classimginf.name, (bx - 100, by + 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            classimginf.location = [bx, by, ex, ey]
            # print(classimginf.prefix)
        else:
            group_mark = impr.clGroupMark_demo()
            group_mark.type = df['type']
            group_mark.name = df['name']
            group_mark.label = df['label']
            group_mark.id = df['id']
            group_mark.data_type = df['data_type']
            dict_option = df['options']
            for dopt in dict_option:
                clmark = impr.clMark_demo()
                clmark.name = dopt['name']
                clmark.label = dopt['label']
                clmark.type = dopt['type']
                clmark.id = dopt['id']
                by = int(dopt['position']['top'])
                bx = int(dopt['position']['left'])
                ex = bx + int(dopt['size']['width'])
                ey = by + int(dopt['size']['height'])
                bx = int(bx * ratiox)
                ex = int(ex * ratiox)
                by = int(by * ratioy)
                ey = int(ey * ratioy)
                cv2.rectangle(img_save_bl, (bx, by), (ex, ey), (0, 0, 255), 4)
                if show_text:
                    cv2.putText(img_save_bl, clmark.name, (bx - 100, by + 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        save_path_rs = os.path.join(save_path, save_filename)
        cv2.imwrite(save_path_rs, img_save_bl)

def store_data_handwriting_template(path, path_config_file, save_path, expand_y=0):
    list_img = impr.list_files1(path, "jpg")
    list_img += impr.list_files1(path, "png")
    dir_name = os.path.dirname(path)

    pred_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
    if save_path is not None:
        result_save_dir = os.path.join(save_path, "aicrhw_" + pred_time)
        if not os.path.isdir(result_save_dir):
            os.mkdir(result_save_dir)
    if len(list_img) == 0:
        print("input image empty")
        return
    template = impr.clTemplate_demo()
    with open(path_config_file, 'r+') as readf:
        count = 0
        for line in readf:
            count += 1
            if count == 1:
                template.name_template = line
            else:
                list_inf = line.split()
                if len(list_inf) == 5:
                    bb_i = [list_inf[0], int(list_inf[1]), int(list_inf[2]), int(list_inf[1]) + int(list_inf[3]),
                            int(list_inf[2]) + int(list_inf[4])]
                    template.listBoxinfor.append(bb_i)
                elif len(list_inf) == 2:
                    template.height = int(list_inf[0])
                    template.width = int(list_inf[1])
    count_img = 0
    list_class_img_info = []
    count = 0
    for path_img in list_img:
        path_img = os.path.join(path, path_img)
        img = cv2.imread(path_img)
        count += 1
        path_save_image = os.path.join(result_save_dir, "AICR_P" + str(count).zfill(7))
        if not os.path.isdir(path_save_image):
            os.mkdir(path_save_image)
        path_org_img = os.path.join(path_save_image, "origine.jpg")
        cv2.imwrite(path_org_img, img)
        img_bl = impr.auto_rotation(img)
        h_n = img_bl.shape[0]
        w_n = img_bl.shape[1]
        ratioy = h_n / template.height
        ratiox = w_n / template.width
        for if_box in template.listBoxinfor:
            prefix = if_box[0]
            count_img += 1
            bx = int(if_box[1] * ratiox)
            by = int((if_box[2] - expand_y) * ratioy)
            ex = int(if_box[3] * ratiox)
            ey = int((if_box[4] + expand_y) * ratioy)
            info_img = impr.crop_image(img_bl, bx, by, ex, ey)
            save_path_img = os.path.join(path_save_image, prefix + ".jpg")
            cv2.imwrite(save_path_img, info_img)
    print("finished !!")


def store_data_handwriting_table(path, save_path, expand_y=0):
    list_img = impr.list_files1(path, "jpg")
    list_img += impr.list_files1(path, "png")
    dir_name = os.path.dirname(path)

    pred_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
    if save_path is not None:
        result_save_dir = os.path.join(save_path, "aicrhw_" + pred_time)
        if not os.path.isdir(result_save_dir):
            os.mkdir(result_save_dir)
    if len(list_img) == 0:
        print("input image empty")
        return
    list_class_img_info = []
    count = 0
    for path_img in list_img:
        path_img = os.path.join(path, path_img)
        img = cv2.imread(path_img)
        count += 1
        path_save_image = os.path.join(result_save_dir, "AICR_P" + str(count).zfill(7))
        if not os.path.isdir(path_save_image):
            os.mkdir(path_save_image)
        path_org_img = os.path.join(path_save_image, "origine.jpg")
        cv2.imwrite(path_org_img, img)
        img_bl = img.copy()
        h_n = img_bl.shape[0]
        w_n = img_bl.shape[1]
        hline_list, vline_list = tbdetect.get_h_and_v_line_bbox_CNX(img_bl)
        list_p_table, hline_list, vline_list = tbdetect.detect_table(hline_list, vline_list)
        count_img = 0
        for table in list_p_table:
            table.detect_cells()
            len_cells = len(table.listCells)
            if len_cells != 12:
                print("error ", path_save_image)
            for id_box in range(len_cells):
                if id_box % 2 != 0:
                    count_img += 1
                    bx, by, ex, ey = table.listCells[id_box]
                    info_img = impr.crop_image(img_bl, bx, by, ex, ey)
                    save_path_img = os.path.join(path_save_image, str(count_img) + ".jpg")
                    cv2.imwrite(save_path_img, info_img)
    print("finished !!")

def convertTXT2JSON(path_txt, path_json):
    list_dict_json = []
    id = 1
    with open(path_txt,mode = 'r+',encoding='utf-8') as readf:
        for line in readf:
            dict_feilds = {}
            dict_pos = {}
            dict_size = {}
            line_str = line.split()
            dict_feilds['id'] = id
            namef, typef, labelf = line_str[0].split('/')
            dict_feilds['name'] = namef
            dict_feilds['type'] = typef
            labelf = labelf.replace('.',' ')
            dict_feilds['label'] = labelf
            dict_pos['left'] = int(line_str[1])
            dict_pos['top'] = int(line_str[2])
            dict_feilds['position'] = dict_pos
            dict_size['width'] = int(line_str[3])
            dict_size['height'] = int(line_str[4])
            dict_feilds['size'] = dict_size
            list_dict_json.append(dict_feilds)
            id += 1
    with open(path_json,mode = 'w',encoding='utf-8') as fout:
        json.dump(list_dict_json, fout,ensure_ascii= False,  indent=2, separators=(',', ': '))