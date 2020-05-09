import os, cv2, codecs, json
import shutil
#from detector.config.config_manager import ConfigManager
import numpy as np
import random

from matplotlib import rc, pyplot as plt
os.environ["PYTHONIOENCODING"] = "utf-8"
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties

classes_vn = "ĂÂÊÔƠƯÁẮẤÉẾÍÓỐỚÚỨÝÀẰẦÈỀÌÒỒỜÙỪỲẢẲẨĐẺỂỈỎỔỞỦỬỶÃẴẪẼỄĨÕỖỠŨỮỸẠẶẬẸỆỊỌỘỢỤỰỴăâêôơưáắấéếíóốớúứýàằầèềìòồờùừỳảẳẩđẻểỉỏổởủửỷãẵẫẽễĩõỗỡũữỹạặậẹệịọộợụựỵ"
class_list_vn = [x for x in classes_vn]

classes_symbol = '*:,@$.-(#%\'\")/~!^&_+={}[]\;<>?※”'
class_list_symbol = [x for x in classes_symbol]

classes_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
class_list_alphabet = [x for x in classes_alphabet]

classes_number = '0123456789'
class_list_number = [x for x in classes_number]

configfile='aicr_dssd_train/config/vietnamese_config.ini'
#configmanager = ConfigManager(configfile)
# img_shape = configmanager.img_shape
# classes = configmanager.classes
# split_overap_size   = configmanager.split_overap_size

img_dir = '/data/CuongND/aicr_vn/data/from_Korea/Vietnamese_TestSet'
file_list = [
    '20190731_144554',
    '20190731_144540',
    '190715070245517_8478000669_pod',
    '190715070249216_8477872491_pod',
    '190715070317353_8479413342_pod'
]

GT_dir = '/data/CuongND/aicr_vn/aicr_dssd_train/outputs/predict_2stages_2019-11-01_09-50'
max_size = 2000
#img_font_idle_size  = configmanager.img_font_idle_size
#img_font_idle_size2 = configmanager.img_font_idle_size2

class Obj_GT:
    def __init__(self, line, scale=1.0):
        data=line.split(" ")
        classID = "vietnam"
        if data[0] in class_list_vn:
            classID = "vietnam"
        elif data[0] in class_list_alphabet:
            classID = "alphabet"
        elif data[0] in class_list_number:
            classID = "number"
        elif data[0] in class_list_symbol:
            classID = "symbol"
        self.class_name = classID
        self.xmin = float(data[1])
        self.ymin = float(data[2])
        self.width = float(data[3])
        self.height = float(data[4])
        self.xmax = self.xmin+self.width
        self.ymax = self.ymin+self.height
        self.is_available=True
    def rescale(self, scale_x, scale_y):
        self.xmin= self.xmin*scale_x
        self.ymin=self.ymin*scale_y
        self.width=self.width*scale_x
        self.height=self.height*scale_y
        self.xmax=self.xmax*scale_x
        self.ymax=self.ymax*scale_y
        return

    def offset(self, x=0,y=0):
        self.xmin=self.xmin-x
        self.ymin=self.ymin-y
        self.line_str=self.class_name+" "+str(int(self.xmin))+" "+str(int(self.ymin))+" "+str(int(self.width))+" "+str(int(self.height))
        return self.line_str

    def export_to_json(self):
        retjson={
            "class_type": self.class_name,
            "char" : '0',
            "x1" : round(self.xmin,6),
            "x2" : round(self.xmax,6),
            "y1" : round(self.ymin,6),
            "y2": round(self.ymax,6)
        }

        return retjson

def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir

def get_list_file_in_folder(dir, ext='png'):
    included_extensions = ['png', 'jpg', 'JPG', 'PNG']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def create_data(GT_file, scale, save_dir, coor_list,img_obj_list, txt_dir='txt', annots_dir='annots', save_img_dir='images'):
    file_name=os.path.basename(GT_file)
    fh1 = codecs.open(GT_file + '.txt', "r", encoding='utf8')
    list_new_bbox=[]

    if not os.path.exists(os.path.join(save_dir,txt_dir)):
        os.makedirs(os.path.join(save_dir,txt_dir))
    if not os.path.exists(os.path.join(save_dir,annots_dir)):
        os.makedirs(os.path.join(save_dir,annots_dir))
    if not os.path.exists(os.path.join(save_dir,save_img_dir)):
        os.makedirs(os.path.join(save_dir,save_img_dir))
    # rescale bbox
    for line in fh1:
        obj=Obj_GT(line.replace("\n", ""))
        obj.rescale(scale,scale)
        list_new_bbox.append(obj)

    for i in range(len(coor_list)):
        list_bbox=[]
        charbox_json_list = []
        is_empty=True
        for j in range(len(list_new_bbox)):
            if(list_new_bbox[j].xmin>=coor_list[i][0] and
                    list_new_bbox[j].ymin >= coor_list[i][1] and
                    list_new_bbox[j].xmin+list_new_bbox[j].width <= coor_list[i][2] and
                    list_new_bbox[j].ymin+list_new_bbox[j].height <= coor_list[i][3] and
                    list_new_bbox[j].is_available):
                list_new_bbox[j].offset(coor_list[i][0], coor_list[i][1])
                list_bbox.append(list_new_bbox[j])
                list_new_bbox[j].is_available=False
                is_empty=False
        result=''
        if is_empty: #don't write if image has no annotation
            continue

        cv2.imwrite(os.path.join(save_dir, save_img_dir, file_name + '_' + str(i) + '.jpg'), img_obj_list[i])

        for bbox in list_bbox:
            result+=bbox.line_str+'\n'
            bbox.rescale(1/320,1/320)
            charbox_json_list.append(bbox.export_to_json())

        with open(os.path.join(save_dir,txt_dir,file_name+'_'+str(i)+'.txt'),'w') as f:
            f.write(result)

        savejson = {"data": charbox_json_list}
        json_filepath=os.path.join(save_dir,annots_dir,file_name+'_'+str(i)+'.json')
        with open(json_filepath, 'w', encoding='utf-8') as fd:
            json.dump(savejson, fd, ensure_ascii=False)

def create_test_set(save_dir='data/test_dir'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for file_name in file_list:
        ori_img_path = os.path.join(img_dir, file_name + '.png')
        print("create test data for image:",ori_img_path)
        if os.path.exists(os.path.join(img_dir, file_name + '.jpg')):
            ori_img_path = os.path.join(img_dir, file_name + '.jpg')
        # load original image with grayscale
        ori_img = cv2.imread(ori_img_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        # 이미지 Grayscale로 변환
        if len(ori_img.shape) > 2:
            ori_img_gray = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
        else:
            ori_img_gray = ori_img.copy()
        print('\nTest image:', ori_img_path, ', original shape(h,w):', ori_img_gray.shape)

        h_, w_ = ori_img_gray.shape
        z_ = 1
        while min(w_, h_) < max_size:
            z__ = 2
            z_ = z_ * 2
            ori_img_gray = cv2.resize(ori_img_gray, (int(w_ * z__), int(h_ * z__)), interpolation=cv2.INTER_LANCZOS4)
            h_, w_ = ori_img_gray.shape
            print('Shape of resize image (h,w):', ori_img_gray.shape, 'zoom_ratio', z_)

        # 문서 내 대다수의 폰트 크기 추정
        img_font_regular_size, avg_character_height = zoomScaleFinder(ori_img_gray, h=float(max_size), AVG_RATIO=0.8,
                                                                      DEBUG=False)
        # Zoom In/Out 비율 획득

        zoom_ratio1 = round(img_font_idle_size / img_font_regular_size, 2)
        print('img_font_regular_size : %.2f' % img_font_regular_size + ' , zoom_ratio : %.2f' % zoom_ratio1)

        # 이미지 리사이즈
        ori_img_gray_resize = cv2.resize(ori_img_gray, None, fx=zoom_ratio1, fy=zoom_ratio1,
                                         interpolation=cv2.INTER_CUBIC)

        # split image for object detection
        img_obj_list, img_coord_list = split_image_to_objs(imgage_obj=ori_img_gray_resize,
                                                           img_shape=img_shape, overap_size=split_overap_size,
                                                           zoom_ratio=zoom_ratio1)
        create_data(os.path.join(GT_dir, file_name),z_*zoom_ratio1,save_dir,img_coord_list, img_obj_list)

def resize_image_and_GT(file_name, image_dir, GT_dir, scale_x=2.5, scale_y=2.5):
    for file in file_name:
        img_path = os.path.join(image_dir, file + '.png')
        GT_path = os.path.join(GT_dir, file + '.txt')

        # rescale ground truth
        fh1 = codecs.open(GT_path, "r", encoding='utf8')
        new_GT = ''
        for line in fh1:
            obj = Obj_GT(line.replace("\n", ""))
            obj.rescale(scale_x, scale_y)
            new_GT += obj.offset() + '\n'
        with open(os.path.join(GT_dir,file+'_resize_'+str(scale_x)+'.txt'), 'w') as f:
            f.write(new_GT)
            print('Save resize GT done')

        # rescale image
        ori_img = cv2.imread(img_path)
        ori_img_upscale = cv2.resize(ori_img, (int(ori_img.shape[1] * scale_x), int(ori_img.shape[0]  * scale_y)), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(image_dir,file+'_resize_'+str(scale_x)+'.png'),ori_img_upscale)
        print ('Save resize image done')

def convert_gray_2RGB(image_dir='', save_dir=''):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    list_img=get_list_file_in_folder(image_dir)

    for img in list_img:
        print (img)
        gray=cv2.imread(os.path.join(image_dir,img))
        #rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(os.path.join(save_dir,img.replace('.jpg','.png')),gray)

def draw_annot_by_text(file_name, image_dir,GT_dir,output_dir,  isGT=True, inch=40, font_path=None, draw_bbox=True):
    for file in file_name:
        print('Draw predicted result to file', file)
        GT_file = os.path.join(GT_dir, file + '.txt')
        img_path = os.path.join(image_dir, file + '.png')

        if os.path.exists(os.path.join(image_dir, file + '.jpg')):
            img_path = os.path.join(image_dir, file + '.jpg')
        content = []
        with codecs.open(GT_file, 'r', encoding='utf8') as f:
            for line in f:
                content.append(line.replace('\n', ''))

        fig, ax = plt.subplots(1)
        fig.set_size_inches(inch, inch)
        ori_img = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)

        plt.imshow(img, cmap='Greys_r')
        for c in content:
            #print(str(c))
            obj = c.split(" ")
            class_nm = obj[0]
            color = 'b'
            conf = 1.0
            if not isGT:
                conf = float(obj[1])
                xmin = int(obj[2])
                ymin = int(obj[3])
                width = int(obj[4])
                height = int(obj[5])
            else:
                xmin = int(obj[1])
                ymin = int(obj[2])
                width = int(obj[3])
                height = int(obj[4])
            if (conf >= 0.8 and conf < 0.9):
                color = 'cyan'
            if(conf>=0.7 and conf<0.8):
                color = 'green'
            if(conf>=0.6 and conf<0.7):
                color = 'orange'
            if(conf>=0.5 and conf<0.6):
                color = 'red'
            if(conf<0.5):
                color = 'black'
            if (height > 0):
                if font_path is None:
                    plt.text(xmin - 2, ymin - 4, class_nm, fontsize=max(int(height / 2), 12), fontdict={"color": 'r'})
                else:
                    prop = FontProperties()
                    prop.set_file(font_path)
                    plt.text(xmin-2, ymin-4, class_nm, fontsize= max(int(height/2), 1), fontdict={"color": 'r'}, fontproperties=prop)
                if not draw_bbox and color=='b':
                    continue
                ax.add_patch(patches.Rectangle((xmin, ymin), width, height,
                                               linewidth=2, edgecolor=color, facecolor='none'))
        # plt.show()
        save_img_path=os.path.join(output_dir,file+'_visualize_result.jpg')
        fig.savefig(save_img_path,bbox_inches='tight')

def count_samples_in_each_class(dir):
    sub_dir_list=get_list_dir_in_folder(dir)
    sub_dir_list=sorted(sub_dir_list)
    count=0
    for cls in class_list_vn:
        if cls not in sub_dir_list:
            print("lack:",cls)

    for cls in class_list_symbol:
        if cls not in sub_dir_list:
            print("lack:",cls)
    for cls in class_list_alphabet:
        if cls not in sub_dir_list:
            print("lack:",cls)
    for cls in class_list_number:
        if cls not in sub_dir_list:
            print("lack:",cls)
    for idx, sub_dir in enumerate(sub_dir_list):
        list_file=get_list_file_in_folder(os.path.join(dir,sub_dir))
        count+=len(list_file)
        print(idx,"Class ",sub_dir,":",len(list_file),"files")
    print('Total: ',count, 'files')

def modify_dataset(dir, remain_percent=0.6):
    sub_dir_list = get_list_dir_in_folder(dir)
    sub_dir_list = sorted(sub_dir_list)
    import random
    for sub_dir in sub_dir_list:
        if sub_dir in class_list_alphabet:
            continue
        list_file = get_list_file_in_folder(os.path.join(dir, sub_dir))
        print("Class ",sub_dir,":",len(list_file),"files before remove")
        total_sample=len(list_file)
        num_file_remain=max(int(remain_percent*total_sample),10000) #keep at least 10000 sample
        num_file_remove=total_sample-num_file_remain
        print("Remove:",num_file_remove, "Remain: ",num_file_remain)
        random.shuffle(list_file)
        for i in range(num_file_remove):
            file_to_remove=os.path.join(dir,sub_dir,list_file[i])
            #print("remove file: ",list_file[i])
            os.remove(file_to_remove)

def filter_result(input_dir, output_dir, list_str="xzpcyvwsôơóốớòồờỏổởõỗỡọộợo"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = get_list_file_in_folder(img_dir)
    file_list = [x.replace('.png', '').replace('.jpg', '') for x in file_list]
    for file_name in file_list:
        result_file = os.path.join(input_dir, file_name + '.txt')
        fh1 = codecs.open(result_file, "r", encoding='utf-8-sig')
        result = ''
        for idx, line in enumerate(fh1):
            data = (line.replace("\n", "")).split(" ")
            if data[0] in list_str:
                line = line.upper()
            result += line
        with open(os.path.join(output_dir, file_name + '.txt'), "w") as f:
            f.write(result)
    print ('Done.')


def drawLabel(data_dir, label, img, drawbox=False):
    print(img)
    plt.rcParams['figure.figsize'] = [100, 50]
    boxes = json.loads(open(label, 'r').read())
    _image = cv2.imread(img)
    _h, _w, _ = _image.shape
    _h = _h*2
    _w=_w*2
    for item in boxes['data']:
#         print(item)
        if drawbox:
            dim = (_w, _h)
            _image = cv2.resize(_image, dim, interpolation = cv2.INTER_AREA)
            cv2.rectangle(_image, (int(item['x1']*_w), int(item['y1']*_h)), (int(item['x2']*_w), int(item['y2']*_h)), (0, 255,0), 2)
        #cv2.rectangle(_image, (int(item['x1']), int(item['y1'])), (int(item['x2']), int(item['y2'])), (0, 255,0), 2)
    plt.imshow(_image)
    plt.title(os.path.basename(img))
    #plt.show()
    plt.savefig(os.path.join(data_dir,os.path.basename(img)), bbox_inches='tight')


def check_anno_image():
    data_dir = '/home/advlab/data/corpus_100_2019-12-03_18-56/'
    arr = range(1, 200)

    annots = sorted(os.listdir(data_dir + '/annots/'))
    images = sorted(os.listdir(data_dir + '/images/'))
    i = 0
    for label, image in zip(annots, images):
        if i in arr:
            # print(label);print(image);print('\n')
            drawLabel(data_dir, data_dir + '/annots/' + label, data_dir + '/images/' + image, True)
        i += 1

def delete_sample(src_dir, dst_dir, num_file=1000):
    list_file=get_list_file_in_folder(src_dir)
    for i in range(len(list_file)-num_file):
        print("delete file:",list_file[i])
        src_file=os.path.join(src_dir, list_file[i])
        os.remove(src_file)

def delete_folder(src_dir, dst_dir):
    list_file=get_list_file_in_folder(src_dir)
    print(src_dir, len(list_file))
    for i in range(100):
        src_file=os.path.join(src_dir, list_file[i])
        dst_file=os.path.join(dst_dir, list_file[i])
        shutil.copy(src_file,dst_file)
    shutil.rmtree(src_dir)

def convert_to_gray(src_dir, dst_dir):
    list_file = get_list_file_in_folder(src_dir)
    print(src_dir, len(list_file))
    for file in list_file:
        print(file)
        src_file=os.path.join(src_dir, file)
        img=cv2.imread(src_file, 0)
        dst_file=os.path.join(dst_dir, file.replace('.png','_gray.jpg'))
        cv2.imwrite(dst_file, img)


def convert_png2jpg(src_dir, dst_dir):
    list_file=get_list_file_in_folder(src_dir)
    for file_name in list_file:
        print(file_name)
        src_path=os.path.join(src_dir,file_name)
        dst_path=os.path.join(dst_dir,file_name.replace('.png','.jpg'))
        img=cv2.imread(src_path)
        cv2.imwrite(dst_path,img)

def check_icdar_sample(img_dir, anno_ext='txt', DB_format=False, num_file=1000):
    list_file=get_list_file_in_folder(img_dir)
    #file_name='bgitem_bg_000511-slant_2-typeface_DejaVuSerifCondensed-BoldItalic-AA_enable_False-weight_0-size_24-rotate_0.png'
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.3
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2
    random.shuffle(list_file)
    print('total:',len(list_file))
    list_file=sorted(list_file)
    for i in range(len(list_file)):
        img_path=os.path.join(img_dir,list_file[i])
        print("draw file:",list_file[i])
        img=cv2.imread(img_path)
        if DB_format:
            anno_path=img_path.replace('png',anno_ext).replace('jpg',anno_ext).replace('JPG',anno_ext).replace('PNG',anno_ext)
        else:
            anno_path=img_path.replace('train_images','train_gts')+'.txt'

        with open(anno_path, "r", encoding='utf-8') as f:
            anno_list = f.readlines()
        anno_list = [x.strip() for x in anno_list]

        for anno in anno_list:
            pts=anno.split(',')
            left=int(pts[0])
            top=int(pts[1])
            right=int(pts[2])
            bottom=int(pts[5])
            loc = -1
            for i in range(0, 8):
                loc = anno.find(',', loc + 1)
            val = anno[loc + 1:]
            img = cv2.rectangle(img, (left,top), (right,bottom), (0,0,255), 2)
            #img = cv2.putText(img,val, (left,top), font, fontScale, color, thickness, cv2.LINE_AA)
            #print(val)

        #cv2.imwrite(img_path.replace('.jpg','_anno.jpg'),img)
        cv2.imshow("res", img)
        cv2.waitKey(0)

def gen_train_txt(img_dir, save_file='train_list.txt'):
    list_file = get_list_file_in_folder(img_dir)
    random.shuffle(list_file)
    print('total:', len(list_file))
    txt=''
    for file in list_file:
        txt+=file+'\n'
    with open(os.path.join(img_dir,'..', save_file), 'w') as f:
        f.write(txt)
    print('Done')


def get_list_file_in_folder(dir, ext='jpg'):
    included_extensions = [ext]
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def convert_jpg2png(src_dir,dst_dir):
    list_file = get_list_file_in_folder(src_dir)
    for file in list_file:
        print(file)
        src_path=os.path.join(src_dir,file)
        dst_path=os.path.join(dst_dir,file.replace('.jpg','.png'))
        img =cv2.imread(src_path)
        cv2.imwrite(dst_path,img)

#typeface_times.png

if __name__== "__main__":
    #get_single_word_from_composed_word('data/corpus/Viet22K.txt')
    #gen_random_serial_corpus()
    #gen_random_number_corpus()
    #gen_random_symbol_corpus()
    #combine_corpus("/data/CuongND/aicr_vn/textimg_data_generator_dev/corpus")
    #os.rename(src_dir+'\nbackground',src_dir+'background')
    transform_image_from_dir()
    #test_background_subtract()
    src='C:/Users/nd.cuong1/PycharmProjects/aicr_vn/textimg_data_generator_dev_vn/data/bg_images2_ori_3'
    dst='C:/Users/nd.cuong1/PycharmProjects/aicr_vn/textimg_data_generator_dev_vn/data/bg_img_jpg'
    #convert_png2jpg(src,dst)
    #gen_train_txt('/home/duycuong/PycharmProjects/research_py3/ss_source/data_generator/outputs/corpus_10000_2020-02-08_18-05/train_images')
    #check_icdar_sample('/home/duycuong/PycharmProjects/research_py3/text_recognition/DB_Liao/datasets/Eval_corpus/train_images', anno_ext='gt')
    src='/home/duycuong/PycharmProjects/research_py3/text_recognition/ss/data_generator/data/bg_img_jpg'
    dst='/home/duycuong/PycharmProjects/research_py3/text_recognition/ss/data_generator/data/bg_img_png'
    #convert_jpg2png(src,dst)