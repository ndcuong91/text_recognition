from __future__ import print_function
from __future__ import division

import argparse
import random, sys
import torch, cv2
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.nn.functional import softmax
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os, time
import models.utils as utils
from utils.loader import ImageFileLoader
import models.crnn as crnn
import models.crnn128 as crnn128
from datetime import datetime
import config_crnn
from torchvision import transforms
from models.utils import writer
from pre_processing.augment_functions import cnd_aug_randomResizePadding, cnd_aug_resizePadding
from torchvision.transforms import ToTensor, Normalize

eval_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
output_dir = 'outputs/eval_' + eval_time
ckpt_prefix = config_crnn.ckpt_prefix
data_dir = config_crnn.train_dir
pretrained = config_crnn.pretrained_test
imgW = config_crnn.imgW
imgH = config_crnn.imgH
gpu = config_crnn.gpu_test
alphabet_path = config_crnn.alphabet_path
workers = config_crnn.workers_test
batch_size = config_crnn.batch_size_test

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
fill_color = (255, 255, 255)  # (209, 200, 193)

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
debug = True

transform_test = transforms.Compose([
    # cnd_aug_randomResizePadding(imgH, imgW, min_scale, max_scale, fill=fill_color, train=False),
    cnd_aug_resizePadding(imgW, imgH, fill=fill_color, train=False),
    ToTensor(),
    Normalize(mean, std)
])

parser = argparse.ArgumentParser()
parser.add_argument('--root', default=data_dir, help='path to root folder')
parser.add_argument('--val', default='val_hw_old.txt', help='path to val set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=workers)
parser.add_argument('--batch_size', type=int, default=batch_size, help='input batch size')
parser.add_argument('--imgH', type=int, default=imgH, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=imgW, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--gpu', default=gpu, help='list of GPUs to use')
parser.add_argument('--alphabet', type=str, default=alphabet_path, help='path to char in labels')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
opt = parser.parse_args()

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
saved = sys.stdout
log_file = os.path.join(output_dir, "eval.log")
f = open(log_file, 'w')
sys.stdout = writer(sys.stdout, f)

print('Please check output of eval process in:', log_file)
print(opt)

if opt.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True
if torch.cuda.is_available() and opt.gpu == None:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

val_dataset = ImageFileLoader(opt.root, opt.val, transform=transform_test)
num_files = len(val_dataset)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=opt.workers,
    shuffle=False
)

alphabet = open(opt.alphabet).read().rstrip()
nclass = len(alphabet) + 1
num_channel = 3

converter = utils.strLabelConverter(alphabet, ignore_case=False)
criterion = CTCLoss()

if opt.imgH == 32:
    crnn = crnn.CRNN32(opt.imgH, num_channel, nclass, opt.nh)
else:
    crnn = crnn.CRNN64(opt.imgH, num_channel, nclass, opt.nh)

# crnn = crnn128.CRNN128(opt.imgH, num_channel, nclass, opt.nh)
if pretrained != '':
    print('loading pretrained model from %s' % pretrained)
    pretrain = torch.load(pretrained)
    crnn.load_state_dict(pretrain, strict=False)

image = torch.FloatTensor(opt.batch_size, num_channel, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batch_size * 5)
length = torch.IntTensor(opt.batch_size)

if opt.gpu is not None:
    crnn.cuda()
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

def decode(converter, preds):
    values, prob = softmax(preds, dim=-1).max(-1)
    preds_idx = (prob > 0).nonzero().squeeze(-1)

    sent_prob = values[preds_idx].mean().item()

    _, preds = preds.max(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))

    preds = preds.view(-1)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return sim_pred, sent_prob

def val(net, data_loader, criterion):
    print('Start val')
    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    val_iter = iter(data_loader)
    val_loss_avg = utils.averager()
    val_cer_avg = utils.averager()
    max_iter = len(data_loader)
    print('Total files:', num_files, ', Number of iters:', max_iter)
    with torch.no_grad():
        for i in range(max_iter):
            if i % 10 == 0:
                print('iter', i)
            data = val_iter.next()
            cpu_images, cpu_texts, imgpath = data
            batch_sz = cpu_images.size(0)
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts)
            utils.loadData(text, t)
            utils.loadData(length, l)

            preds = net(image)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_sz))
            cost = criterion(preds, text, preds_size, length) / batch_sz
            cost = cost.detach().item()
            val_loss_avg.add(cost)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            if batch_size == 1:
                sim_preds = [sim_preds]
            cer_loss = utils.cer_loss(sim_preds, cpu_texts)
            if debug and cer_loss[0] > 0 and batch_size == 1:
                print(i,'\nimg path', imgpath)
                print('sim pred', sim_preds)
                print('cpu text', cpu_texts)
                print('cer', cer_loss)
                inv_tensor = inv_normalize(cpu_images[0])
                cv_img = inv_tensor.permute(1, 2, 0).numpy()
                cv_img_convert = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                cv2.imshow('image data', cv_img_convert)
                ch = cv2.waitKey(0)
                if ch == 27:
                    break
            val_cer_avg.add(cer_loss)

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('\nraw: %-30s \nsim: %-30s\n gt: %-30s' % (raw_pred, pred, gt))
    test_loss = val_loss_avg.val()
    test_cer = val_cer_avg.val()
    print('\nTest loss: %f - test cer %f' % (test_loss, test_cer))
    return test_loss, test_cer


begin_val = time.time()
val(crnn, val_loader, criterion)
end_val = time.time()
processing_time = end_val - begin_val
print('Processing time:', processing_time)
print('Speed:', num_files / processing_time, 'fps')

sys.stdout = saved
f.close()
