# common
imgW = 1600
imgH = 64
alphabet_path = 'data/char_238'

# train
train_dir = '/data/train_data_29k_29Feb_update30Mar'
pretrained = ''
# pretrained='outputs/train_2020-02-21_13-44_train_ocr_dataset/AICR_pretrained_30_Test_3.808_cer_0.0269.pth'
pretrained = 'outputs/train_2020-04-11_10-02_train_ocr_char_238/AICR_train_ocr_dataset_10April_new_augment_char_238_64_39_loss_3.7_cer_0.0266.pth'
#pretrained='ckpt/AICR_finetune_new_data_44_loss_7.266_cer_0.028.pth'
gpu_train = '0'  # '0,1' or None
base_lr = 0.0005
max_epoches = 500
workers_train = 8
batch_size = 64
train_file = 'train_merge.txt'
val_file = 'val_merge.txt'
ckpt_prefix = 'AICR_finetune_char_238_64'

# test
test_dir = '/home/aicr/cuongnd/data/test/GWP/crop'
pretrained_test = 'outputs/train_2020-04-15_11-20/AICR_finetune_char_238_64_239_loss_0.69_cer_0.0053.pth'
#pretrained_test = 'ckpt/AICR_SDV_30Mar_300_loss_1.25_cer_0.0076.pth'
# for testing SDV printing character cases
label = False
test_list = ''

gpu_test = '1'
#gpu_test = None
workers_test = 8
batch_size_test = 1
debug = False
if debug:
    batch_size_test = 1
