import torchvision
import torch
import os
import torch.utils.data as data
from PIL import Image
try:
    from models.utils import resizePadding
except ImportError:
    from classifier_CRNN.models.utils import resizePadding

def default_flist_reader(root, flist):
    imlist = []
    img_exts = ('jpg', 'png', 'JPG', 'PNG')
    if flist!='':
        with open(os.path.join(root,flist)) as rf:
            for line in rf.readlines():
                impath = line.strip()
                if impath.endswith(img_exts):
                    imlabel = os.path.splitext(impath)[0] + '.txt'
                    imlist.append((impath, imlabel))
    else:
        file_names = [fn for fn in os.listdir(root)
                      if any(fn.endswith(ext) for ext in img_exts)]
        file_names = sorted(file_names)
        for file in file_names:
            imlabel = os.path.splitext(file)[0] + '.txt'
            imlist.append((file, imlabel))
    return imlist

class alignCollate(object):
    def __init__(self, imgW, imgH):
        self.imgH = imgH
        self.imgW = imgW

    def __call__(self, batch):
        images, labels, img_paths = zip(*batch)
        images = [resizePadding(image, self.imgW, self.imgH) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        return images, labels, img_paths

class ImageFileLoader(data.Dataset):
    def __init__(self, root,  flist = '', flist_reader = default_flist_reader, transform=None, label=True):
        self.root = root
        self.imlist = flist_reader(root,flist)
        self.transform=transform
        self.label=label

    def __getitem__(self, index):
        impath, labelpath = self.imlist[index]
        imgpath = os.path.join(self.root, impath)
        labelpath = os.path.join(self.root, labelpath)
        img = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = ''
        if self.label:
            label = open(labelpath.replace('/images/', '/annos/')).read().rstrip('\n')
        return img, label, imgpath

    def __len__(self):
        return len(self.imlist)

class NumpyListLoader(data.Dataset):  #no label
    def __init__(self, numpylist, transform=None):
        self.imlist = numpylist
        self.transform = transform

    def __getitem__(self, index):
        imdata = self.imlist[index]
        img = Image.fromarray(imdata).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, '', ''

    def __len__(self):
        return len(self.imlist)

def main():
    pass

if __name__ == '__main__':
    main()
