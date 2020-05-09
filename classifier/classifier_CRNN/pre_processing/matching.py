import cv2, os, json
import numpy as np
import imutils
from matplotlib import pyplot as plt

# img = cv2.imread(dst, 0)
# img2 = img.copy()
# _template_ = cv2.imread(src, 0)
#w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

#this matches for logo
#methods = ['cv2.TM_CCORR_NORMED']
#methods = ['cv2.TM_CCOEFF_NORMED']

def doRotate(img, degree, scale=1.0):
    (w, h) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, degree, scale)
    return cv2.warpAffine(img, M, (h, w))

def doMatch(_template_, _dst, scale = 1, angle = 10, meth='cv2.TM_CCORR_NORMED', _show = True, debug = False):
    t_img = cv2.imread(_template_)
    t_i = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY)
    t_img = cv2.adaptiveThreshold(t_i, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 41, 7)
    #t_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY)
    #template = imutils.resize(doRotate(t_img, angle), width=int(t_img.shape[1]*scale))
    template = imutils.resize(t_img, width=int(t_img.shape[1]*scale))
    img_o = cv2.imread(_dst)
    _img = imutils.resize(cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY), width=1600)
    img = cv2.adaptiveThreshold(_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 41, 7)
    w, h = template.shape[::-1]
    #img2 = img.copy()#doRotate(img, angle)
    #img = imutils.resize(img2, width = int(img2.shape[1] * scale))
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    if (debug):
        print(_dst, _template_, max_val, min_val, scale, angle, meth)
    if _show:
        cv2.rectangle(img_o, top_left, bottom_right, 255, 2)
        cv2.imshow("detected image", img_o)
        cv2.waitKey(0)
        # plt.subplot(121),plt.imshow(res,cmap = 'gray')
        # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122),plt.imshow(img,cmap = 'gray')
        # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        # plt.suptitle(meth)
        # plt.show()

    return top_left, bottom_right, max_val, min_val

def matchArea(src, dst, _meth = ['cv2.TM_CCOEFF_NORMED'], f_log = "error.log", show=False, debug=False):
    error_log = open(f_log, "a")
    mx, my = (0, 0)
    mval = 0.0
    msc = 0.0
    mrot = 0.0
    minval = 0.0
    rot = 0
    for smeth in _meth:
        #for rot in np.linspace(-3, 3, num=5):
            for sc in np.linspace(1.0, 3.0, num=5):
                try:
                    cx, cy, val, _minval = doMatch(src, dst, scale=sc, angle=rot, meth=smeth, _show=False, debug=False)
                    if mval < val:
                        mrot = rot
                        mval = val
                        mx = cx
                        my = cy
                        msc = sc
                        _meth = smeth
                        minval = _minval
                except Exception as e:
                    print("template is {} and image is {}".format(src, dst), file=error_log)
                    print("angle is {} and scale is {}".format(rot, sc), file=error_log)
                    print("used method is {}".format(smeth), file=error_log)
                    print("exception occured! error is {}".format(e), file=error_log)
                    continue
    error_log.close()
    if debug:
        print(dst, src, smeth, msc, mrot, mval, minval)
        print(mx)
        print(my)
    if (mx[0]>0 and show==True):
        img = cv2.imread(dst, 0)
        #img1 = doRotate(img, mrot)
        #img2 = imutils.resize(img1, width = int(img.shape[1] * msc))
        cv2.rectangle(img, mx, my, 255, 2)
        cv2.imshow("detected image", img)
        cv2.waitKey(0)
    return mx, my, msc, mval, minval

def findCorners(imPath, _dir = "data/id-template", templates=["logo.jpg", "area-I.jpg", "area-II.jpg", "area-III.jpg", "area-IV.jpg"]):
    #templates=["area-I.jpg"]
    img = imutils.resize(cv2.imread(imPath), width=1600)
    for tpl in templates:
        px, py, sc, _max, _min = matchArea(os.path.join(_dir, tpl), imPath, debug=True)
        '''tx = px
        ty = py
        tx[0] = int(px[0] * sc)
        tx[1] = int(px[1] * sc)
        ty[0] = int(py[0] * sc)
        ty[1] = int(py[1] * sc)'''
        if (_max > 0.45):
            cv2.rectangle(img, px, py, 255, 2)
        #cv2.circle(img, px, 20, (255, 0, 0), 2)
    _img = imutils.resize(img, width=800)
    # plt.subplot(122), plt.imshow(img, cmap = 'gray')
    # plt.title('Detected Area'), plt.xticks([]), plt.yticks([])
    # plt.suptitle("detected area")
    # plt.show()
    cv2.imshow("detected image", _img)
    cv2.waitKey(0)

def loadTemplate(jfile):
    _dir = "data/SDV_invoices"
    listdir = ["I-1.png", "II-2.png", "II-3.png", "II-4.png"]
    content = json.load(open(jfile))
    anchor = []
    _t = ''
    for item in content["types"]:
        print(item["size"])
        # print(item["template-dir"])
        _t = item["template-dir"]
        anchor.append([anc["file"] for anc in item["data"]])
        #print(anchor[0])
    for _f in listdir: findCorners(os.path.join(_dir, _f), _dir=_t, templates=anchor[0])
    return

# I-1.jpg logo.jpg logo-I.jpg logo-II.jpg logo-IIa.jpg logo-III.jpg logo-IIIa.jpg logo-II-60.jpg logo-II-30.jpg
# I-2a.jpg 
# _src = os.path.join(_dir, "I-2.jpg")
# _dir = "data/IDcard/CMND_old_1"
# listdir = ["28.jpg"]
# listdir = os.listdir(_dst)
# listdir.sort()
# _dir = "data/SDV"
# "I-1.png", "II-2.png", "II-3.png", "II-4.png"
# "II-5.png", II-6.png, II-7.png
# listdir = ["I-1.png", "II-2.png", "II-3.png", "II-4.png"]
# listdir = os.listdir(_dir)
# for _f in listdir:
#     findCorners(os.path.join(_dir, _f), _dir="data/template", templates=["ky-hieu.png", "mau-so.png"])
    #matchArea(_src, os.path.join(_dst, _f))

loadTemplate("data/template/anchor_config.json")