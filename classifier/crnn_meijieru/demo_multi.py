import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import time, os, cv2

import models.crnn as crnn


model_path = '/home/duycuong/PycharmProjects/research_py3/text_recognition/crnn_pbcquoc/outputs/netCRNN_43.pth'
img_dir='../EAST/outputs/predict_Eval_model.ckpt-45451/SCAN_20191128_145142994_002'
#img_path = './data/demo.png'
alphabet = open('/home/duycuong/PycharmProjects/research_py3/text_recognition/crnn_pbcquoc/data_icdar/char').read().rstrip()
debug = False

def get_list_file_in_folder(dir, ext='png'):
    included_extensions = ['png', 'jpg']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

model = crnn.CRNN(32, 3, 229, 256)
# if torch.cuda.is_available():
#     model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet, ignore_case=False)
transformer = dataset.resizeNormalize((100, 32))

begin = time.time()

list_files=get_list_file_in_folder(img_dir)
total_file=len(list_files)
print('Predict:',total_file,'files')

for file in list_files:
    img_path=os.path.join(img_dir,file)
    image = Image.open(img_path).convert('L')
    image = transformer(image)
    # if torch.cuda.is_available():
    #     image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    #print(sim_pred)
    if debug:
        img = cv2.imread(img_path)
        cv2.imshow(sim_pred,img)
        cv2.waitKey(0)


#raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
#print('%-20s => %-20s' % (raw_pred, sim_pred))
end = time.time()
processing_time=end-begin
print('Processing time:',processing_time)
print('Speed:', total_file/processing_time, 'fps')
