import torch
from torch.autograd import Variable
from models.utils import strLabelConverter
import models.crnn as crnn
import models.crnn128 as crnn128
from torch.nn.functional import softmax
import time, os, cv2
from utils.loader import ImageFileLoader, alignCollate, NumpyListLoader
import models.utils as utils
import config_crnn
from torchvision import transforms
from pre_processing.augment_functions import cnd_aug_randomResizePadding, cnd_aug_resizePadding
from torchvision.transforms import RandomApply, ColorJitter, RandomAffine, ToTensor, Normalize

img_dir = config_crnn.test_dir
test_list = config_crnn.test_list
pretrained = config_crnn.pretrained_test
imgW = config_crnn.imgW
imgH = config_crnn.imgH
gpu = config_crnn.gpu_test
alphabet_path = config_crnn.alphabet_path
workers = config_crnn.workers_test
batch_size = config_crnn.batch_size_test
label = config_crnn.label
debug = True
alphabet = open(alphabet_path).read().rstrip()
nclass = len(alphabet) + 1
nc = 3
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
fill_color = (255, 255, 255) #(209, 200, 193)
min_scale, max_scale = 2 / 3, 2

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

transform_test = transforms.Compose([
    #cnd_aug_randomResizePadding(imgH, imgW, min_scale, max_scale, fill=fill_color, train=False),
    cnd_aug_resizePadding(imgW, imgH, fill=fill_color, train=False),
    ToTensor(),
    Normalize(mean, std)
])

def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir

def decode(converter, preds):
    values, prob = softmax(preds, dim=-1).max(-1)
    preds_idx = (prob > 0).nonzero().squeeze(-1)

    sent_prob = values[preds_idx].mean().item()

    _, preds = preds.max(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))

    preds = preds.view(-1)
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return raw_pred, sim_pred, sent_prob

def predict(dir, batch_sz, max_iter=10000):
    print('Init CRNN classifier')
    image = torch.FloatTensor(batch_sz, 3, imgH, imgH)
    model = crnn.CRNN64(imgH, nc, nclass, 256)
    #model = crnn128.CRNN128(imgH, nc, nclass, 256)
    if gpu != None:
        print('Use GPU', gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        model = model.cuda()
        image = image.cuda()
    print('loading pretrained model from %s' % pretrained)
    model.load_state_dict(torch.load(pretrained, map_location='cpu'))

    converter = strLabelConverter(alphabet, ignore_case=False)
    val_dataset = ImageFileLoader(dir, flist=test_list, label=label, transform=transform_test)
    num_files = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_sz,
        num_workers=workers,
        shuffle=False
    )

    image = Variable(image)

    # for p in crnn.parameters():
    #     p.requires_grad = False
    model.eval()
    print('Start predict in folder',img_dir)
    val_iter = iter(val_loader)
    max_iter = min(max_iter, len(val_loader))
    print('Number of samples', num_files)
    begin = time.time()
    with torch.no_grad():
        for i in range(max_iter):
            data = val_iter.next()
            cpu_images, cpu_texts, img_paths = data
            batch_size = cpu_images.size(0)
            utils.loadData(image, cpu_images)
            preds = model(image)

            preds = preds.squeeze(1)
            raw_pred, sim_pred, sent_prob = decode(converter, preds)

            # preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            # _, preds = preds.max(2)
            # preds = preds.transpose(1, 0).contiguous().view(-1)
            # sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
            # #print(sim_pred)
            # raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
            #print(cpu_texts[0])
            if debug:
                print('\n', raw_pred)
                print('\n',round(sent_prob,3), sim_pred, img_paths)
                inv_tensor = inv_normalize(cpu_images[0])
                cv_img = inv_tensor.permute(1, 2, 0).numpy()
                cv_img_convert = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                cv2.imshow('image data', cv_img_convert)
                ch = cv2.waitKey(0)
                if ch == 27:
                    break
    end = time.time()
    processing_time = end - begin
    print('Processing time:', processing_time)
    print('Speed:', num_files / processing_time, 'fps')


if __name__ == "__main__":
    list_dir = get_list_dir_in_folder(img_dir)
    if len(list_dir)>0:
        for subdir in list_dir:
            predict(os.path.join(img_dir, subdir), batch_size)
    else:
        predict(img_dir, batch_size)
