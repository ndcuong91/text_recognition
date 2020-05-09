import os, shutil, random


def get_list_file_in_folder(dir, ext='png'):
    included_extensions = ['jpg', 'png', 'JPG', 'PNG']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names


def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir


def prepare_train_DB_from_data_generator(data_dir, train_ratio=0.95):
    train_imgs = os.path.join(data_dir, 'train_images')
    train_gts = os.path.join(data_dir, 'train_gts')
    test_imgs = os.path.join(data_dir, 'test_images')
    test_gts = os.path.join(data_dir, 'test_gts')
    try:
        os.makedirs(test_imgs)
        os.makedirs(test_gts)
        shutil.move(os.path.join(data_dir, 'images'), train_imgs)
        shutil.move(os.path.join(data_dir, 'annots'), train_gts)
    except:
        pass

    list_files = get_list_file_in_folder(train_imgs)
    random.shuffle(list_files)
    num_trainval = len(list_files)
    num_train = int(train_ratio * num_trainval)
    num_val = num_trainval - num_train
    print('train',num_train,'val',num_val)
    train_txt = ''
    test_txt = ''
    count = 0
    for idx, file in enumerate(list_files):
        if(idx<num_val):
            print(idx,'move', list_files[idx])
            src_img = os.path.join(train_imgs, list_files[idx])
            dst_img = os.path.join(test_imgs, list_files[idx])
            src_anno = os.path.join(train_gts, list_files[idx] + '.txt')
            dst_anno = os.path.join(test_gts, list_files[idx] + '.txt')
            shutil.move(src_img, dst_img)
            shutil.move(src_anno, dst_anno)
            test_txt += list_files[idx] + '\n'
        else:
            train_txt+=list_files[idx]+'\n'

    with open(os.path.join(data_dir, 'train_list.txt'), 'w', encoding='utf-8') as f:
        f.write(train_txt)
    with open(os.path.join(data_dir, 'test_list.txt'), 'w', encoding='utf-8') as f:
        f.write(test_txt)
    print('Total word:', count)


if __name__ == "__main__":
    data_dir = '/home/aicr/cuongnd/ss_source/data_generator/outputs/corpus_1000_2020-03-11_16-32'
    prepare_train_DB_from_data_generator(data_dir)
