#!python3
import argparse
import os
import torch
import cv2
import numpy as np
import math, time
from structure.model import SegDetectorModel
from structure.representers.seg_detector_representer import SegDetectorRepresenter
from structure.visualizers.seg_detector_visualizer import SegDetectorVisualizer

img_path = '../../datasets/detector/invoices_7May/imgs_crop/val/350.jpg'
detector_model = 'model_epoch_428_minibatch_9000'
ckpt_path = '../detector_DB_train/outputs/workspace/detector_DB_train/outputs/train_2020-05-10_22-50/model/' + detector_model
# detector_model = 'pre-trained-model-synthtext-resnet18'
# ckpt_path = '/home/aicr/cuongnd/aicr.core/detector_DB_train/pretrained/' + detector_model

polygon = False
visualize = True
img_short_side = 960  # 736 960
detector_box_thres = 0.01
gpu_test = -1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_test)


def main():
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint', default=ckpt_path)
    parser.add_argument('--result_dir', type=str, default='./demo_results/', help='path to save results')
    parser.add_argument('--data', type=str,
                        help='The name of dataloader which will be evaluated on.')
    parser.add_argument('--image_short_side', type=int, default=img_short_side,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--thresh', type=float,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--box_thresh', type=float, default=detector_box_thres,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--resize', action='store_true', help='resize')
    parser.add_argument('--visualize', default=visualize, help='visualize maps in tensorboard')
    parser.add_argument('--polygon', help='output polygons if true', default=polygon)
    parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
                        help='Show iamges eagerly')

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    detector = Demo(gpu=gpu_test, cmd=args)
    begin = time.time()
    detector.inference(img_path, args['visualize'])
    end = time.time()
    print('Inference time:', end - begin, 'seconds')


class Demo:
    def __init__(self, gpu='0', cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.gpu = gpu
        self.args = cmd
        self.init_torch_tensor()
        self.init_model(self.args['resume'])
        self.model.eval()

        self.segRepresent = SegDetectorRepresenter()
        self.segVisualizer = SegDetectorVisualizer()

    def init_torch_tensor(self):
        if torch.cuda.is_available() and self.gpu != None:
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

    def init_model(self, path):
        self.model = SegDetectorModel(self.device, distributed=False, local_rank=0)
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(path, map_location=self.device)
        self.model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img

    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape

    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i, :, :].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")

    def inference(self, image_path, visualize=False):
        batch = dict()
        batch['filename'] = [image_path]
        img, original_shape = self.load_image(image_path)
        batch['shape'] = [original_shape]
        with torch.no_grad():
            batch['image'] = img
            pred = self.model.forward(batch, training=False)
            output = self.segRepresent.represent(batch, _pred=pred, is_output_polygon=self.args['polygon'])
            if not os.path.isdir(self.args['result_dir']):
                os.mkdir(self.args['result_dir'])
            self.format_output(batch, output)
            boxes, _ = output
            boxes = boxes[0]

            if visualize:
                vis_image = self.segVisualizer.demo_visualize(image_path, output)
                cv2.imwrite(os.path.join(self.args['result_dir'],
                                         image_path.split('/')[-1].split('.')[0] + '_ ' + detector_model + '_ ' + str
                                         (detector_box_thres) + '.jpg'), vis_image)
            return boxes


if __name__ == '__main__':
    main()
