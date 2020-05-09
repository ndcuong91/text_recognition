import os, cv2, random
import time
import numpy as np
import shutil
from utils.unicode_utils import compound_unicode

icdar_dir = '/home/aicr/cuongnd/aicr.core/data_generator/outputs/corpus_400_2020-03-30_14-25/images'
output_dir = '/home/aicr/cuongnd/aicr.core/data_generator/outputs/corpus_400_2020-03-30_14-25/CRNN_SDV_train'


def get_list_file_in_folder(dir, ext='png'):
    included_extensions = ['jpg', 'png', 'JPG', 'PNG']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir

def crop_from_img_rectangle(img, left, top, right, bottom):
    extend_y = max(int((bottom - top) / 3), 4)
    extend_x = int(extend_y / 2)
    top = max(0, top - extend_y)
    bottom = min(img.shape[0], bottom + extend_y)
    left = max(0, left - extend_x)
    right = min(img.shape[1], right + extend_x)
    if left >= right or top >= bottom or left < 0 or right < 0 or left >= img.shape[1] or right >= img.shape[1]:
        return None
    return img[top:bottom, left:right]

def prepare_train_from_icdar(data_dir, output_dir, same_folder=True):
    try:
        os.makedirs(os.path.join(output_dir, 'images'))
        os.makedirs(os.path.join(output_dir, 'annos'))
    except:
        pass

    list_files = get_list_file_in_folder(data_dir)
    train_txt = ''
    test_txt = ''
    max_wh = 0
    count = 0
    for idx, file in enumerate(list_files):
        # if(idx>1000):
        #     continue
        print(idx, file)
        base_name = file.replace('.png', '')
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path)
        anno_path = img_path.replace('.png', '.txt')
        with open(anno_path, 'r', encoding='utf-8') as f:
            anno_list = f.readlines()
        anno_list = [x.strip() for x in anno_list]
        for index, anno in enumerate(anno_list):
            pts = anno.split(',')
            left = int(pts[0])
            top = int(pts[1])
            right = int(pts[2])
            bottom = int(pts[5])
            loc = -1
            for i in range(0, 8):
                loc = anno.find(',', loc + 1)
            val = anno[loc + 1:]
            crop = crop_from_img_rectangle(img, left, top, right, bottom)
            if crop is None or val == '':
                continue
            if ((crop.shape[1] / crop.shape[0]) > max_wh):
                max_wh = crop.shape[1] / crop.shape[0]
            count += 1
            base_crop_name = base_name + '_' + str(index)
            if (random.randint(0, 4) == 0):
                test_txt += 'images/' + base_crop_name + '.jpg\n'
            else:
                train_txt += 'images/' + base_crop_name + '.jpg\n'
            crop_img_path = os.path.join(output_dir, 'images', base_crop_name + '.jpg')
            crop_anno_path = os.path.join(output_dir, 'annos', base_crop_name + '.txt')
            cv2.imwrite(crop_img_path, crop)
            if same_folder:
                crop_anno_path = crop_img_path.replace('.jpg','.txt')
            with open(crop_anno_path, 'w', encoding='utf-8') as f:
                f.write(val)

    with open(os.path.join(output_dir, 'train'), 'w', encoding='utf-8') as f:
        f.write(train_txt)
    with open(os.path.join(output_dir, 'test'), 'w', encoding='utf-8') as f:
        f.write(test_txt)
    print('max width height ratio in dataset', max_wh)
    print('Total word:', count)


def prepare_train_test_from_multiple_dir(root_dir, list_dir, percentage=1.0, train_ratio=1.0, convert=False):
    list_dir_txt = []
    print('root dir:', root_dir)

    alphabet = open('/home/aicr/cuongnd/aicr.core/classifier_CRNN/data/char_238').read().rstrip()
    for dir in list_dir:
        # if os.path.exists(os.path.join(root_dir,dir,'origine.jpg')):
        #     os.remove(os.path.join(root_dir,dir,'origine.jpg'))
        #     print('remove file',os.path.join(root_dir,dir,'origine.jpg'))
        list_files = get_list_file_in_folder(os.path.join(root_dir, dir))
        # convert_json_to_multiple_gt(os.path.join(root_dir, dir))
        print('Dir ', dir, 'has', len(list_files), 'files')
        for idx, file in enumerate(list_files):
            #print(file)
            # fix unicode
            txt_file=os.path.join(root_dir,dir, file).replace('.jpg','.txt').replace('.png','.txt')
            with open(txt_file, 'r', encoding='utf8', errors='ignore') as f:
                txt= f.readlines()
            if len(txt)>0:
                new_text=compound_unicode(txt[0])
                for ch in txt[0]:
                    if ch not in alphabet:
                        print (ch, file)
                        #os.remove(os.path.join(root_dir,dir, file))
                        continue
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(new_text)
            list_dir_txt.append(os.path.join(dir, file))

    random.shuffle(list_dir_txt)
    print('\ntotal files:', len(list_dir_txt))
    num_files_to_use = int(percentage * len(list_dir_txt))
    num_files_to_train = int(train_ratio * num_files_to_use)
    num_files_to_test = num_files_to_use - num_files_to_train
    print('\ntrain files:', num_files_to_train)
    print('\ntest files:', num_files_to_test)

    train_list = list_dir_txt[0:num_files_to_train - 1]
    test_list = list_dir_txt[num_files_to_train:num_files_to_use - 1]

    print('Write train test file')
    train_txt = ''
    for train_file in train_list:
        train_txt += train_file + '\n'
    with open(os.path.join(root_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write(train_txt)

    test_txt = ''
    for test_file in test_list:
        test_txt += test_file + '\n'
    with open(os.path.join(root_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.write(test_txt)

    print('Done')

def fix_unicode(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as f:
        txt = f.readlines()
    final_str=''
    for anno in txt:
        final_str+=anno
    if len(txt) > 0:
        new_text = compound_unicode(final_str)
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(new_text)


def prepare_txt_file(data_dir):
    list_files = get_list_file_in_folder(data_dir)
    save_txt = ''
    for file in list_files:
        save_txt += os.path.join(data_dir, file) + '\n'
        with open(os.path.join(data_dir, file.replace('.jpg', '.txt').replace('.png', '.txt')), 'w',
                  encoding='utf-8') as f:
            f.write('abc')
    with open(os.path.join(data_dir + '/..', 'test'), 'w', encoding='utf-8') as f:
        f.write(save_txt)


def convert_json_to_multiple_gt(dir, json_name='labels.json'):
    import json
    with open(os.path.join(dir, json_name)) as json_file:
        data = json.load(json_file)
        for key, value in data.items():
            gt_name = key.replace('.jpg', '.txt').replace('.png', '.txt')
            with open(os.path.join(dir, gt_name), 'w', encoding='utf-8') as f:
                f.write(value)


def crop_collected_data(dir, file_list=['2', '8', '14', '20'], debug=False):
    list_files = get_list_file_in_folder(dir)
    for file in list_files:
        if file.replace('.jpg', '') in file_list:
            file_path = os.path.join(dir, file)
            print(file_path)
            ori_img = cv2.imread(file_path)
            crop_img = ori_img[0:82, 0:ori_img.shape[1]]
            if debug:
                cv2.imshow('result', crop_img)
                cv2.waitKey(0)
            cv2.imwrite(file_path, crop_img)


def crop_collected_data2(dir, debug=False):
    list_files = get_list_file_in_folder(dir)
    count1 = 0
    count2 = 0
    for file in list_files:
        file_path = os.path.join(dir, file)
        print(file_path)
        ori_img = cv2.imread(file_path)
        if ori_img.shape[1] / ori_img.shape[0] > 9:
            count1 += 1
            extend_val = int(ori_img.shape[0] / 9) + 3
            print('count1', count1, 'extend val', extend_val, 'height', ori_img.shape[0])
            crop_img = ori_img[extend_val:ori_img.shape[0] - extend_val - 1, 0:ori_img.shape[1]]
            if debug:
                cv2.imshow('ttt', crop_img)
                cv2.waitKey(0)
            cv2.imwrite(file_path, crop_img)
        elif ori_img.shape[1] / ori_img.shape[0] > 7:
            count2 += 1
            extend_val = int(ori_img.shape[0] / 9) + 1
            print('count2', count2, 'extend val', extend_val, 'height', ori_img.shape[0])
            crop_img = ori_img[extend_val:ori_img.shape[0] - extend_val - 1, 0:ori_img.shape[1]]
            if debug:
                cv2.imshow('result', crop_img)
                cv2.waitKey(0)
            cv2.imwrite(file_path, crop_img)


def create_val_from_collected_data(src_dir, dst_dir):
    list_dir = get_list_dir_in_folder(src_dir)
    count = 0
    for idx, dir in enumerate(list_dir):
        count += 1
        print(count, dir)
        if (idx < 65):
            continue
        name1 = str(random.randint(1, 24))
        new_name1 = dir[-4:] + '_' + name1
        src_file1 = os.path.join(src_dir, dir, name1)
        dst_file1 = os.path.join(dst_dir, new_name1)
        print('Move', src_file1)
        print('To', dst_file1)
        if os.path.exists(src_file1 + '.jpg') and os.path.exists(os.path.exists(src_file1 + '.txt')):
            shutil.move(src_file1 + '.jpg', dst_file1 + '.jpg')
            shutil.move(src_file1 + '.txt', dst_file1 + '.txt')

        name2 = str(random.randint(1, 24))
        new_name2 = dir[-4:] + '_' + name2
        src_file2 = os.path.join(src_dir, dir, name2)
        dst_file2 = os.path.join(dst_dir, new_name2)
        print('Move', src_file2)
        print('To', dst_file2)
        if os.path.exists(src_file2 + '.jpg') and os.path.exists(os.path.exists(src_file2 + '.txt')):
            shutil.move(src_file2 + '.jpg', dst_file2 + '.jpg')
            shutil.move(src_file2 + '.txt', dst_file2 + '.txt')


def gen_blank_image(target_dir, num=150):
    for i in range(num):
        h = random.randint(32, 150)
        w = random.randint(int(h / 2), int(10 * h))
        print(i, w, h)

        blank_img = np.zeros([h, w, 3], dtype=np.uint8)
        blank_img.fill(255)
        cv2.imwrite(os.path.join(target_dir, str(i) + '.jpg'), blank_img)
        with open(os.path.join(target_dir, str(i) + '.txt'), 'w') as f:
            f.write('')

def remove_unsuitable_samples(dir, alphabet_path='../data/char_238'):
    alphabet = open(alphabet_path, encoding='utf8').read().rstrip()
    list_file = get_list_file_in_folder(dir)
    count=0
    for file in list_file:
        anno_path = os.path.join(dir,file.replace('.jpg','.txt').replace('.png','.txt'))
        anno_txt = open(anno_path, encoding='utf8').read().rstrip()
        error=False
        for ch in anno_txt:
            if ch not in alphabet:
                print(ch)
                error = True
        if error:
            print (count)
            print(file)
            print(anno_txt)
            print()
            count += 1
            os.remove(os.path.join(dir,file))
            os.remove(anno_path)


if __name__ == "__main__":
    remove_unsuitable_samples('/data/dataset/ocr_dataset/random')
    #prepare_train_from_icdar(icdar_dir, output_dir)
    #fix_unicode('/data/train_data_29k_29Feb_update30Mar/val_printed')

    final_list_dir=['random', 'InkData_line_processed']
    root_dir = '/data/dataset/ocr_dataset'

    # data_dir='cleaned_data_merge_fixed/train'
    # list_dir=get_list_dir_in_folder(os.path.join(root_dir,data_dir))
    # for dir in list_dir:
    #     final_list_dir.append(os.path.join(data_dir,dir))
    #
    #
    # data_dir='cinnamon_data/train'
    # list_dir=get_list_dir_in_folder(os.path.join(root_dir,data_dir))
    # for dir in list_dir:
    #     final_list_dir.append(os.path.join(data_dir,dir))
    #
    # final_list_dir.append('blank_images')

    #data_dir='cleaned_data_30Mar/test'
    # list_dir=get_list_dir_in_folder(os.path.join(root_dir,data_dir))
    # for dir in list_dir:
    #     final_list_dir.append(os.path.join(data_dir,dir))

    # final_list_dir.append('cinnamon_data/cinamon_test_115')
    #
    # data_dir='cleaned_data_merge_fixed/AICR_test1'
    # list_dir=get_list_dir_in_folder(os.path.join(root_dir,data_dir))
    # for dir in list_dir:
    #     final_list_dir.append(os.path.join(data_dir,dir))
    #
    # final_list_dir.append('cleaned_data_merge_fixed/AICR_test2')

    #final_list_dir.append('printed_30Mar')

    prepare_train_test_from_multiple_dir(root_dir, final_list_dir, percentage=1.0, train_ratio=0.9)
    # img_dir='/home/duycuong/PycharmProjects/dataset/ocr_dataset/meta'
    # crop_collected_data2(img_dir)
    # gen_blank_image('/home/duycuong/PycharmProjects/dataset/blank_images')
    # create_val_from_collected_data('/home/duycuong/PycharmProjects/dataset/cleaned_data_merge_fixed','/home/duycuong/PycharmProjects/dataset/AICR_test2')
    # for dir in list_dir:
    #     crop_collected_data(os.path.join(root_dir,dir))
    # kk=1
    # prepare_train_from_icdar(
    #     "/data/generated_data_sdv/corpus_500_2020-03-25_18-04/images",
    #     "/data/generated_data_sdv/crnn-test"
    # )
