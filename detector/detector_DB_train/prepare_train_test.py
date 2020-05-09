import os, shutil, random
import pathlib


def get_list_file_in_folder(dir, ext=['jpg', 'png', 'JPG', 'PNG']):
    included_extensions = ext
    # included_extensions = ['txt']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names


def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir


def rename_files(dir):
    list_file = get_list_file_in_folder(dir, 'txt')
    for file in list_file:
        os.rename(os.path.join(dir, file), os.path.join(dir, file.replace('.jpg.txt', '.txt')))


def prepare_train_DB_from_data_generator(data_dir, train_ratio=0.95):
    train_imgs = os.path.join(data_dir, 'train_images')
    train_gts = os.path.join(data_dir, 'train_gts')
    test_imgs = os.path.join(data_dir, 'test_images')
    test_gts = os.path.join(data_dir, 'test_gts')
    # try:
    #     #os.makedirs(test_imgs)
    #     #os.makedirs(test_gts)
    #     #shutil.move(os.path.join(data_dir, 'images'), train_imgs)
    #     #shutil.move(os.path.join(data_dir, 'annots'), train_gts)
    # except:
    #     pass

    list_files = get_list_file_in_folder(train_imgs)
    random.shuffle(list_files)
    num_trainval = len(list_files)
    num_train = int(train_ratio * num_trainval)
    num_val = num_trainval - num_train
    print('train', num_train, 'val', num_val)
    train_txt = ''
    test_txt = ''
    count = 0
    for idx, file in enumerate(list_files):
        if (idx < num_val):
            print(idx, 'move', list_files[idx])
            src_img = os.path.join(train_imgs, list_files[idx])
            dst_img = os.path.join(test_imgs, list_files[idx])
            src_anno = os.path.join(train_gts, list_files[idx].split('.')[0] + '.txt')
            dst_anno = os.path.join(test_gts, list_files[idx].split('.')[0] + '.txt')
            shutil.move(src_img, dst_img)
            shutil.move(src_anno, dst_anno)
            test_txt += list_files[idx] + '\n'
        else:
            train_txt += list_files[idx] + '\n'

    with open(os.path.join(data_dir, 'train_list.txt'), 'w', encoding='utf-8') as f:
        f.write(train_txt)
    with open(os.path.join(data_dir, 'test_list.txt'), 'w', encoding='utf-8') as f:
        f.write(test_txt)
    print('Total word:', count)


def convertVOCtoDB(text_dirs, outp):
    '''
    :param text_dirs: list of VOC label file directories
    :param outp: output directory will be contained DB label
    :return: None
    '''
    if not os.path.exists(outp):
        os.makedirs(outp)
    text_pathes = []
    for source_path in text_dirs:
        for filepath in pathlib.Path(source_path).glob('**/*'):
            filename, file_extension = os.path.splitext(str(filepath))
            if os.path.isfile(str(filepath)) and file_extension.lower() in ['.txt']:
                text_pathes.append(str(filepath))
    # for file in filelist:

    for filepath in text_pathes:
        filename, file_extension = os.path.splitext(str(filepath))
        filename = filename.split("/")[-1]
        # print(filename)
        if filename != 'classes':
            with open(filepath, 'r', encoding='utf-8') as f:
                objBoxes = f.readlines()
            # print(objBoxes)
            dbObjBoxes = []
            for box in objBoxes:
                box = box.replace('\n', '')
                box = box.split()
                label, x, y, w, h = box
                # print(label, x, y, w, h)
                x1 = str(int(x) + int(w))
                y1 = str(int(y) + int(h))
                newBox = ','.join([x, y, x1, y, x1, y1, x, y1, label])
                dbObjBoxes.append(newBox)
                # db box: x0,y0,x1,y0,x1,y1,x0,y1,label
            str_newboxes = '\n'.join(dbObjBoxes)
            with open(os.path.join(outp, filename + file_extension), 'w', encoding='utf-8') as f:
                f.write(str_newboxes)

    # print(text_pathes)


def convert_ICDAR_to_VOC(text_dirs, outp):
    '''
    :param text_dirs: list of VOC label file directories
    :param outp: output directory will be contained DB label
    :return: None
    '''
    if not os.path.exists(outp):
        os.makedirs(outp)
    text_pathes = []
    for source_path in text_dirs:
        for filepath in pathlib.Path(source_path).glob('**/*'):
            filename, file_extension = os.path.splitext(str(filepath))
            if os.path.isfile(str(filepath)) and file_extension.lower() in ['.txt']:
                text_pathes.append(str(filepath))
    # for file in filelist:

    for filepath in text_pathes:
        filename, file_extension = os.path.splitext(str(filepath))
        filename = filename.split("/")[-1]
        print(filename)
        if filename != 'class':
            with open(filepath, 'r', encoding='utf-8') as f:
                objBoxes = f.readlines()
            # print(objBoxes)
            dbObjBoxes = []
            for box in objBoxes:
                box = box.replace('\n', '')
                box = box.split(',')
                label = box[8]
                if box[8] == 'label':
                    label = 'text'
                w = int(box[2]) - int(box[0])
                h = int(box[5]) - int(box[1])
                newBox = ' '.join([label, box[0], box[1], str(w), str(h)])
                dbObjBoxes.append(newBox)
                # db box: x0,y0,x1,y0,x1,y1,x0,y1,label
            str_newboxes = '\n'.join(dbObjBoxes)
            with open(os.path.join(outp, filename + file_extension), 'w', encoding='utf-8') as f:
                f.write(str_newboxes)
            kk = 1

    # print(text_pathes)


if __name__ == "__main__":
    # rename_files('/home/aicr/cuongnd/aicr.core/detector_DB_train/datasets/invoices_28April/train_gts')
    data_dir = '/home/aicr/cuongnd/aicr.core/detector_DB_train/datasets/invoices_28April'
    # prepare_train_DB_from_data_generator(data_dir)

    convert_ICDAR_to_VOC(['D:/invoices_28April/train_gts'], 'D:/invoices_28April/train_gts')
    # convertVOCtoDB(['/data/data_label/data_final'],'/data/data_label/data_anno_DB')
