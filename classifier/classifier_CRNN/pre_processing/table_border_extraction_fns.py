import cv2, math, logging, numpy as np


kernel_sharpening = np.array([[0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [-1, -1, -1, -1, 17, -1, -1, -1, -1],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0]])

kernel_blur = np.array([[1 / 9, 1 / 9, 1 / 9],
                        [1 / 9, 1 / 9, 1 / 9],
                        [1 / 9, 1 / 9, 1 / 9]])

dl = logging.getLogger("debug")


class clTable:
    def __init__(self):
        self.startX = -1
        self.startY = -1
        self.endX = -1
        self.endY = -1
        self.idTable = -1
        self.listVertical = []
        self.listHorizontal = []
        self.listPoints = []
        self.listCells  = list()
    def __lt__(self, other):
        return ( ( self.endX - self.startX ) * ( self.endY - self.startY ) )\
               < ( ( other.endX - other.startX ) * ( other.endY - other.startY ) )
    def detect_list_points(self,expand = 10):
        self.listPoints.clear()
        point = [self.startX,self.startY]
        self.listPoints.append(point)
        point = [self.startX,self.endY]
        self.listPoints.append(point)
        point = [self.endX, self.endY]
        self.listPoints.append(point)
        point = [self.endX, self.startY]
        self.listPoints.append(point)
        for lh in self.listHorizontal:
            for lv in self.listVertical:
                if lh[3] <= (lv[3] + expand) and lh[3] >= (lv[1] - expand) \
                        and lv[0] <= (lh[2] + expand) and lv[0] >= (lh[0] - expand):
                    point = [lv[0],lh[3]]
                    if point not in self.listPoints:
                        self.listPoints.append(point)

    def check_same_Hor_or_Ver(self,p1,p2,checkHor = True, expand = 10):
        if checkHor == True:
            line_check = None
            for h in self.listHorizontal:
                if abs(p1[1] - h[1]) <= expand and p1[0] >= ( h[0] - expand ) and p1[0] <= ( h[2] + expand):
                    if abs(p2[1] - h[1]) <= expand and p2[0] >= ( h[0] - expand ) and p2[0] <= ( h[2] + expand):
                        return True
            return False
        else:
            line_check = None
            for v in self.listVertical:
                if abs(p1[0] - v[0]) <= expand and p1[1] >= ( v[1] - expand ) and p1[1] <= ( v[3] + expand ):
                    line_check = v
                    if abs(p2[0] - v[0]) <= expand and p2[1] >= (v[1] - expand) and p2[1] <= (v[3] + expand):
                        return True
            return False

    def detect_cells(self):
        self.listCells.clear()
        self.detect_list_points()
        numb_p = len(self.listPoints)
        for i in range(numb_p):
            topleft = self.listPoints[i]
            rs_topr = list(filter(lambda x: x[1] == topleft[1] and  x[0] > topleft[0],self.listPoints))
            rs_topr.sort()
            check_out = False
            for topright in rs_topr:
                if self.check_same_Hor_or_Ver(topleft,topright):
                    rs_bottomr = list(filter(lambda x: x[0] == topright[0] and  x[1] > topright[1],self.listPoints))
                    rs_bottomr.sort()
                    for bottomright in rs_bottomr:
                        if self.check_same_Hor_or_Ver(topright,bottomright,False):
                            bottomleft = [topleft[0],bottomright[1]]
                            if bottomleft in self.listPoints:
                                if self.check_same_Hor_or_Ver(bottomright,bottomleft) \
                                        and self.check_same_Hor_or_Ver(topleft,bottomleft,False):
                                    if bottomright[1] - topleft[1] < 5 or bottomright[0] - topleft[0]< 5:
                                        continue
                                    cell = [topleft[0],topleft[1],bottomright[0],bottomright[1]]
                                    self.listCells.append(cell)
                                    check_out = True
                        if check_out == True:
                            break
                if check_out == True:
                    break
        self.listCells.sort(key=lambda x:(x[1],x[0]))
def detect_table_from_image(img, detect_cell = True):
    h = img.shape[0]
    hline_list, vline_list = get_h_and_v_line_bbox_CNX(img)
    list_p_table, hline_list, vline_list = detect_table(hline_list, vline_list, int(h / 20))
    if detect_cell == True:
        for i in range(0, len(list_p_table)):
            list_p_table[i].detect_cells()
    return list_p_table, hline_list, vline_list

def detect_table(h_p_info, v_p_info, minsizetable=10, distance_p=10, distance_connect = 10 ):
    h_p_info_filter = filter_lines_error(h_p_info, False)
    v_p_info_filter = filter_lines_error(v_p_info)

    sorted(h_p_info_filter,key= lambda x:x[1])
    sorted(v_p_info_filter,key = lambda x:x[0])
    v_p_info_filter, h_p_info_filter = reconnect_lines(v_p_info_filter,h_p_info_filter)
    len_h = len(h_p_info_filter)
    len_v = len(v_p_info_filter)
    # print(len_h)
    index_h = [0] * len_h
    index_v = [0] * len_v
    id_table = 0
    list_table = []
    for v in range(len_v):

        type_infor = [0]*4
        idH_infor = [-1]*4
        idP_infor = [[-1,-1]]*4

        bgV = [v_p_info_filter[v][0], v_p_info_filter[v][1]]
        eV = [v_p_info_filter[v][2], v_p_info_filter[v][3]]
        wV = eV[1] - bgV[1]

        for h in range(len_h):
            bgH = [h_p_info_filter[h][0], h_p_info_filter[h][1]]
            eH = [h_p_info_filter[h][2], h_p_info_filter[h][3]]
            wH = eH[0] - bgH[0]

            dis_min = distance_p
            type_t = -1
            # type -1 none table 0 topleft, 1 topright, 2 bottomleft, 3 bottomright
            valx  = abs(bgV[0] - bgH[0])
            valy = abs(bgV[1] - bgH[1])
            if valx <= dis_min and valy <= dis_min:
                type_t = 0
            else:
                valx = abs(bgV[0] - eH[0])
                valy = abs(bgV[1] - eH[1])
                if valx <= dis_min and valy <= dis_min:
                    type_t = 1
                else:
                    valx = abs(eV[0] - bgH[0])
                    valy = abs(eV[1] - bgH[1])
                    if valx <= dis_min and valy <= dis_min:
                        type_t = 2
                    else:
                        valx = abs(eV[0] - eH[0])
                        valy = abs(eV[1] - eH[1])
                        if valx <= dis_min and valy <= dis_min:
                            type_t = 3
                        else:
                            type_t = -1
            if type_t != -1:
                if type_infor[type_t] == 1:
                    if (valx + valy) < (idP_infor[type_t][0] + idP_infor[type_t][1]):
                        idP_infor[type_t][0] = valx
                        idP_infor[type_t][1] = valy
                        idH_infor[type_t] = h
                else:
                    idP_infor[type_t][0] = valx
                    idP_infor[type_t][1] = valy
                    idH_infor[type_t] = h
                    type_infor[type_t] = 1
        for i in range(4):
            if type_infor[i] == 0:
                continue
            h = idH_infor[i]
            bgH = [h_p_info_filter[h][0], h_p_info_filter[h][1]]
            eH = [h_p_info_filter[h][2], h_p_info_filter[h][3]]
            if index_v[v] == 0 and index_h[h] == 0:
                table = clTable()
                list_table.append(table)
                id_table = len(list_table)
                index_h[h] = id_table
                index_v[v] = id_table
                table.idTable = id_table
            elif index_v[v] != 0:
                id_table = index_v[v]
                index_h[h] = index_v[v]
            else:
                id_table = index_h[h]
                index_v[v] = index_h[h]
            if i == 0:
                list_table[id_table - 1].startX = bgV[0]
                list_table[id_table - 1].startY = bgH[1]
            elif i == 1:
                list_table[id_table - 1].startY = eH[1]
                list_table[id_table - 1].endX   = bgV[0]
            elif i == 2:
                list_table[id_table - 1].endY = bgH[1]
                list_table[id_table - 1].startX = eV[0]
            else:
                list_table[id_table - 1].endX = eV[0]
                list_table[id_table - 1].endY = eH[1]
    result_tb = []
    list_idtable = []
    list_table.sort()
    for t in list_table:
        if t.endY != -1 and t.startX != -1 and t.endX != -1 and t.startY != -1:
            if (t.endY - t.startY ) > minsizetable or  (t.endX - t.startX) > minsizetable:
                result_tb.append(t)
                list_idtable.append(t.idTable)

    index_h = [0] * len_h
    index_v = [0] * len_v

    expand_size = 20
    for t in result_tb:
        id_table = t.idTable
        for h in range(len_h):
            if index_h[h] == 0:
                coor_y = h_p_info_filter [h][1]
                coor_x_s = h_p_info_filter[h][0] + expand_size
                coor_x_e = h_p_info_filter[h][2] - expand_size
                if coor_y <= t.endY and coor_y >= t.startY and coor_x_e <= t.endX and coor_x_s >= t.startX:
                    index_h[h] = id_table
                    t.listHorizontal.append(h_p_info_filter[h])
        for v in range(len_v):
            if index_v[v] == 0:
                coor_x = v_p_info_filter[v][0]
                coor_y_s = v_p_info_filter[v][1] + expand_size
                coor_y_e = v_p_info_filter[v][3] - expand_size
                if coor_x <= t.endX and coor_x >= t.startX and coor_y_e <= t.endY and coor_y_s >= t.startY:
                    index_v[v] = id_table
                    t.listVertical.append(v_p_info_filter[v])
    result_tb2 = []
    for t in result_tb:
        if len(t.listHorizontal) >= 2 and len(t.listVertical) > 0:
            result_tb2.append(t)
    for t in result_tb2:
        len_v = len(t.listVertical)
        len_h = len(t.listHorizontal)
        sorted(t.listVertical, key=lambda x: x[3])
        sorted(t.listHorizontal, key=lambda x: x[2])
        processed_v = [0] * len_v
        processed_h = [0] * len_h
        list_v_after_filter = []
        list_h_after_filter = []
        for v in range(len_v):
            if processed_v[v] == 1:
                continue
            processed_v[v] = 1
            bVx = t.listVertical[v][0]
            bVy = t.listVertical[v][1]
            eVx = t.listVertical[v][2]
            eVy = t.listVertical[v][3]
            for v2 in range(v+1,len_v):
                if processed_v[v2] == 1:
                    continue
                bVx1 = t.listVertical[v2][0]
                bVy1 = t.listVertical[v2][1]
                eVx1 = t.listVertical[v2][2]
                eVy1 = t.listVertical[v2][3]
                if abs(bVx - bVx1) < distance_connect:
                    e_check = min(eVy,eVy1)
                    b_check = max(bVy,bVy1)
                    bCombinate = True
                    for h in t.listHorizontal:
                        ycheck = h[1]
                        xhb = h[0]
                        xhe = h[2]
                        if ycheck <= b_check and ycheck >= e_check and bVx > xhb and bVx < xhe:
                            bCombinate = False
                            break
                    if bCombinate == True:
                        eVy = max(eVy,eVy1)
                        bVy = min(bVy,bVy1)
                        processed_v[v2] = 1
            list_v_after_filter.append([bVx,bVy,eVx,eVy])

        for h in range(len_h):
            if processed_h[h] == 1:
                continue
            processed_h[h] = 1
            bHx = t.listHorizontal[h][0]
            bHy = t.listHorizontal[h][1]
            eHx = t.listHorizontal[h][2]
            eHy = t.listHorizontal[h][3]
            for h2 in range(h + 1,len_h):
                if processed_h[h2] == 1:
                    continue
                bHx1 = t.listHorizontal[h2][0]
                bHy1 = t.listHorizontal[h2][1]
                eHx1 = t.listHorizontal[h2][2]
                eHy1 = t.listHorizontal[h2][3]
                if abs(bHy - bHy1) < distance_connect:
                    e_check = min(eHx,eHx1)
                    b_check = max(bHx,bHx1)
                    bCombinate = True
                    for h in t.listVertical:
                        xcheck = h[0]
                        yvb    = h[1]
                        yve    = h[3]
                        if xcheck <= b_check and xcheck >= e_check and eHy > yvb and eHy < yve:
                            bCombinate = False
                            break
                    if bCombinate == True:
                        eHx = max(eHx,eHx1)
                        bHx = min(bHx,bHx1)
                        processed_h[h2] = 1
            list_h_after_filter.append([bHx,bHy,eHx,eHy])
        t.listVertical = list_v_after_filter
        t.listHorizontal = list_h_after_filter
    result_tb2.sort(key=lambda x: ( x.startY,x.startX))
    return result_tb2, h_p_info_filter, v_p_info_filter

def reconnect_lines(v_lines, h_lines, distance_offset = 15):
    len_v = len(v_lines)
    len_h = len(h_lines)
    sorted(v_lines, key=lambda x: x[3])
    sorted(h_lines, key=lambda x: x[2])
    processed_v = [0] * len_v
    processed_h = [0] * len_h
    list_v_after_filter = []
    list_h_after_filter = []
    for v in range(len_v):
        if processed_v[v] == 1:
            continue
        processed_v[v] = 1
        bVx = v_lines[v][0]
        bVy = v_lines[v][1]
        eVx = v_lines[v][2]
        eVy = v_lines[v][3]
        for v2 in range(v + 1, len_v):
            if processed_v[v2] == 1:
                continue
            bVx1 = v_lines[v2][0]
            bVy1 = v_lines[v2][1]
            eVx1 = v_lines[v2][2]
            eVy1 = v_lines[v2][3]
            if abs(bVx - bVx1) < distance_offset:
                e_check = min(eVy, eVy1)
                b_check = max(bVy, bVy1)
                if abs(e_check - b_check ) < distance_offset :
                    bCombinate = True
                    if abs(e_check - b_check ) > 5:
                        for h in h_lines:
                            ycheck = h[1]
                            xhb = h[0]
                            xhe = h[1]
                            if ycheck <= b_check and ycheck >= e_check and xhb < eVx and xhe > eVx:
                                bCombinate = False
                                break
                    if bCombinate == True:
                        eVy = max(eVy, eVy1)
                        bVy = min(bVy, bVy1)
                        processed_v[v2] = 1
        list_v_after_filter.append([bVx, bVy, eVx, eVy])

    for h in range(len_h):
        if processed_h[h] == 1:
            continue
        processed_h[h] = 1
        bHx = h_lines[h][0]
        bHy = h_lines[h][1]
        eHx = h_lines[h][2]
        eHy = h_lines[h][3]
        for h2 in range(h + 1, len_h):
            if processed_h[h2] == 1:
                continue
            bHx1 = h_lines[h2][0]
            bHy1 = h_lines[h2][1]
            eHx1 = h_lines[h2][2]
            eHy1 = h_lines[h2][3]
            if abs(bHy - bHy1) < distance_offset:
                e_check = min(eHx, eHx1)
                b_check = max(bHx, bHx1)
                if abs(e_check - b_check) < distance_offset:
                    bCombinate = True
                    if abs(e_check - b_check) > 5:
                        for h in v_lines:
                            xcheck = h[0]
                            yvb = h[1]
                            yve = h[3]
                            if xcheck <= b_check and xcheck >= e_check and eHy > yvb and eHy <yve :
                                bCombinate = False
                                break
                    if bCombinate == True:
                        eHx = max(eHx, eHx1)
                        bHx = min(bHx, bHx1)
                        processed_h[h2] = 1
        list_h_after_filter.append([bHx, bHy, eHx, eHy])
    return list_v_after_filter, list_h_after_filter

def filter_lines_error(list_lines, isVertical=True, error_pos=20):
    len_l = len(list_lines)
    visited_l = [0] * len_l
    result_affter_filter = []
    for h in list_lines:
        if isVertical == True:
            if h[0] < error_pos or h[1] < error_pos or h[2] < error_pos or h[3] < error_pos:
                continue
            if abs(h[0] - h[2]) <= error_pos and abs(h[1] - h[3]) >= error_pos:
                h[0] = (h[0] + h[2]) / 2
                h[2] =  h[0]
                result_affter_filter.append(h)
        else:
            if h[0] < error_pos or h[1] < error_pos or h[2] < error_pos or h[3] < error_pos:
                continue
            if abs(h[1] - h[3]) <= error_pos and abs(h[0] - h[2]) >= error_pos:
                h[1] = (h[1] + h[3]) / 2
                h[3] = h[1]
                result_affter_filter.append(h)
    return result_affter_filter


def get_h_and_v_line_bbox_CNX(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(gray_img, -1, kernel_sharpening)

    kernal_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    clahe_applied_img = clahe.apply(gray_img)

    # sharpened = cv2.filter2D(clahe_applied_img, -1, kernel_sharpening)
    # cv2.imwrite("sharpe2.jpg", sharpened)

    adaptivethreshold_applied_img = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                          cv2.THRESH_BINARY, 27, 7)
    clahe_applied_img = clahe.apply(adaptivethreshold_applied_img)

    inverted_img = 255 - adaptivethreshold_applied_img
    clahe_applied_img = clahe.apply(inverted_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vertical_extraction_iter_1_img = cv2.morphologyEx(inverted_img, cv2.MORPH_ERODE, kernel, iterations=1)

    vertical_extraction_iter_5_img = cv2.morphologyEx(inverted_img, cv2.MORPH_ERODE, kernel, iterations=3)
    vertical_extraction_buffup_iter_5_img = cv2.morphologyEx(vertical_extraction_iter_5_img, cv2.MORPH_DILATE, kernel,
                                                             iterations=3)
    #cv2.imwrite("verr.jpg", vertical_extraction_buffup_iter_5_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    horizontal_extraction_img = cv2.morphologyEx(inverted_img, cv2.MORPH_ERODE, kernel, iterations=3)

    horizontal_extraction_buffup_iter_5_img = cv2.morphologyEx(horizontal_extraction_img, cv2.MORPH_DILATE, kernel,
                                                               iterations=3)
    #cv2.imwrite("hor.jpg",horizontal_extraction_buffup_iter_5_img)
    _, _, horizontal_line_info, _ = cv2.connectedComponentsWithStats(horizontal_extraction_buffup_iter_5_img, 4)

    # ignore the first one
    horizontal_line_info_list = horizontal_line_info[1:]

    horizontal_line_info_list[:, 2] = horizontal_line_info_list[:, 0] + horizontal_line_info_list[:, 2]
    horizontal_line_info_list[:, 3] = horizontal_line_info_list[:, 1] + horizontal_line_info_list[:, 3]

    _, _, vertical_line_info, _ = cv2.connectedComponentsWithStats(vertical_extraction_buffup_iter_5_img, 4)

    vertical_line_info_list = vertical_line_info[1:]

    vertical_line_info_list[:, 2] = vertical_line_info_list[:, 0] + vertical_line_info_list[:, 2]
    vertical_line_info_list[:, 3] = vertical_line_info_list[:, 1] + vertical_line_info_list[:, 3]

    horizontal_line_info_list = horizontal_line_info_list[:, :4]
    vertical_line_info_list = vertical_line_info_list[:, :4]

    return horizontal_line_info_list, vertical_line_info_list


def get_h_and_v_line_bbox(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=5.5, tileGridSize=(8, 8))
    clahe_applied_img = clahe.apply(gray_img)

    adaptivethreshold_applied_img = cv2.adaptiveThreshold(clahe_applied_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                          cv2.THRESH_BINARY, 41, 7)

    inverted_img = 255 - adaptivethreshold_applied_img

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    vertical_extraction_iter_1_img = cv2.morphologyEx(inverted_img, cv2.MORPH_ERODE, kernel, iterations=1)

    vertical_extraction_iter_5_img = cv2.morphologyEx(inverted_img, cv2.MORPH_ERODE, kernel, iterations=5)

    vertical_extraction_buffup_iter_5_img = cv2.morphologyEx(vertical_extraction_iter_5_img, cv2.MORPH_DILATE, kernel,
                                                             iterations=4)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    horizontal_extraction_img = cv2.morphologyEx(inverted_img, cv2.MORPH_ERODE, kernel, iterations=5)

    horizontal_extraction_buffup_iter_5_img = cv2.morphologyEx(horizontal_extraction_img, cv2.MORPH_DILATE, kernel,
                                                               iterations=4)

    _, _, horizontal_line_info, _ = cv2.connectedComponentsWithStats(horizontal_extraction_buffup_iter_5_img, 4)

    # ignore the first one
    horizontal_line_info_list = horizontal_line_info[1:]

    horizontal_line_info_list[:, 2] = horizontal_line_info_list[:, 0] + horizontal_line_info_list[:, 2]
    horizontal_line_info_list[:, 3] = horizontal_line_info_list[:, 1] + horizontal_line_info_list[:, 3]

    # draw horizontal lines on top of original image

    # copyimg = img.copy()
    # for line_info in horizontal_line_info_list:
    #     cv2.line(copyimg, (line_info[0], line_info[1]) , (line_info[2], line_info[3]) , (0,0,255), 2 )

    # saveimg_as(copyimg, "horizontal_line_draw")

    _, _, vertical_line_info, _ = cv2.connectedComponentsWithStats(vertical_extraction_buffup_iter_5_img, 4)

    vertical_line_info_list = vertical_line_info[1:]

    vertical_line_info_list[:, 2] = vertical_line_info_list[:, 0] + vertical_line_info_list[:, 2]
    vertical_line_info_list[:, 3] = vertical_line_info_list[:, 1] + vertical_line_info_list[:, 3]

    # copyimg = img.copy()
    # for line_info in vertical_line_info_list:
    #     cv2.line(copyimg, (line_info[0], line_info[1]) , (line_info[2], line_info[3]) , (0,0,255), 2 )

    horizontal_line_info_list = horizontal_line_info_list[:, :4]
    vertical_line_info_list = vertical_line_info_list[:, :4]

    return horizontal_line_info_list, vertical_line_info_list


def dummy_force_get_hline_end_to_end_points(hline_info):
    p1 = hline_info[:2]
    p2 = hline_info[2:]

    return p1, p2


def dummy_force_get_vline_end_to_end_points(vline_info):
    p1 = vline_info[:2]
    p2 = vline_info[2:]

    return p1, p2


def force_strict_centerpoint_horizontal_end_to_end_points(hline_info):
    cy = (hline_info[1] + hline_info[3]) / 2
    cy = int(cy)

    p1 = [hline_info[0], cy]
    p2 = [hline_info[2], cy]

    return p1, p2


def convert_hinfo_strict_centerpoint_horizontal(hline_info):
    cy = (hline_info[1] + hline_info[3]) / 2
    cy = int(cy)

    x1 = min(hline_info[0], hline_info[2])
    x2 = max(hline_info[0], hline_info[2])

    new_hline = [x1, cy, x2, cy]

    return new_hline


def force_strict_centerpoint_vertical_end_to_end_points(vline_info):
    cx = (vline_info[0] + vline_info[2]) / 2
    cx = int(cx)

    p1 = [cx, vline_info[1]]
    p2 = [cx, vline_info[3] - 1]

    return p1, p2


def convert_vinfo_strict_centerpoint_vertical(vline_info):
    cx = (vline_info[0] + vline_info[2]) / 2
    cx = int(cx)

    y1 = min(vline_info[1], vline_info[3])
    y2 = max(vline_info[1], vline_info[3])

    new_vline = [cx, y1, cx, y2]

    return new_vline


def _get_inverse_slope_of_lineinfo(lineinfo):
    delta_x = lineinfo[2] - lineinfo[0]
    delta_y = lineinfo[3] - lineinfo[1]

    if delta_y == 0:
        return 99999
    else:
        return float(delta_x) / delta_y


def _get_slope_of_lineinfo(lineinfo):
    delta_x = lineinfo[2] - lineinfo[0]
    delta_y = lineinfo[3] - lineinfo[1]

    if delta_x == 0:
        return 99999
    else:
        return float(delta_y) / delta_x


def _check_non_inclusive_overlap_horizontally(lineinfo1, lineinfo2):
    base_x1 = lineinfo1[0]
    base_x2 = lineinfo1[2]

    test_x1 = lineinfo2[0]
    test_x2 = lineinfo2[2]

    if test_x1 > base_x1 and test_x1 < base_x2:
        return True

    if test_x2 > base_x1 and test_x2 < base_x2:
        return True

    return False


def _check_non_inclusive_overlap_vertically(lineinfo1, lineinfo2):
    base_y1 = lineinfo1[1]
    base_y2 = lineinfo1[3]

    test_y1 = lineinfo2[1]
    test_y2 = lineinfo2[3]

    if test_y1 > base_y1 and test_y1 < base_y2:
        return True

    if test_y2 > base_y1 and test_y2 < base_y2:
        return True

    return False


def _check_no_overlap_horizontally(lineinfo1, lineinfo2):
    q1 = lineinfo1[0]
    q2 = lineinfo1[2]

    s1 = lineinfo2[0]
    s2 = lineinfo2[2]

    overlap = max(0, min(q2, s2) - max(q1, s1))

    if overlap > 0:
        return False
    else:
        return True


def _check_no_overlap_vertically(lineinfo1, lineinfo2):
    q1 = lineinfo1[1]
    q2 = lineinfo1[3]

    s1 = lineinfo2[1]
    s2 = lineinfo2[3]

    overlap = max(0, min(q2, s2) - max(q1, s1))

    if overlap > 0:
        return False
    else:
        return True


def _calc_length_between_two_points(p1, p2):
    diff_x = p1[0] - p2[0]
    diff_y = p1[1] - p2[1]

    length = math.sqrt(diff_x ** 2 + diff_y ** 2)

    return length


def _get_gap_length(lineinfo1, lineinfo2):
    gap_length1 = _calc_length_between_two_points(lineinfo1[:2], lineinfo2[2:])
    gap_length2 = _calc_length_between_two_points(lineinfo1[2:], lineinfo2[:2])

    return min(gap_length1, gap_length2)


def _get_length_of_lineinfo(lineinfo):
    return _calc_length_between_two_points(lineinfo[:2], lineinfo[2:])


def clm_horizontal_lines_v1(line_info_list, slope_diff_threshold=0.5, gap_percentage_threshold=0.05):
    """
    do continuous line merge with given line infos and return the merged results

    """

    # calculate slopes of all line_info_list

    slope_list = []

    for lineinfo in line_info_list:
        slope = _get_slope_of_lineinfo(lineinfo)
        slope_list.append(slope)

    merge_index_pair_list = []

    for i in range(len(line_info_list)):
        for j in range(i + 1, len(line_info_list)):
            line_info_1 = line_info_list[i]
            line_info_2 = line_info_list[j]

            # if the lines are not "non-inclusive overlap" mode, then there is no possibility of merge

            if not _check_no_overlap_horizontally(line_info_1, line_info_2):

                if not _check_non_inclusive_overlap_horizontally(line_info_1, line_info_2):
                    dl.debug("not non_inclusive_overlap_horizontally")
                    continue
                else:
                    dl.debug("non inclusive overlap horizontally pass!")

            slope_1 = slope_list[i]
            slope_2 = slope_list[j]

            # evaluate is slope 1 and 2 are similar (within threshold)

            slope_diff = abs(slope_2 - slope_1)

            if slope_diff > slope_diff_threshold:
                continue

            # evaluate if the gap is small
            gap_length = _get_gap_length(line_info_1, line_info_2)
            gap_length_threshold = gap_percentage_threshold * min(_get_length_of_lineinfo(line_info_1),
                                                                  _get_length_of_lineinfo(line_info_2))
            dl.debug("gap_length={} , gap_length_threshold: {}".format(gap_length, gap_length_threshold))

            if gap_length > gap_length_threshold:
                dl.debug("gap length too large")
                continue
            else:
                dl.debug("gap length pass!")

            pair = (i, j)
            merge_index_pair_list.append(pair)

    dl.debug("merge_index_pair_list: {}".format(merge_index_pair_list))

    ## do merge

    # first union the pairs into groups

    merge_index_group_list = []

    dl.debug("merge_index_group_list: {}".format(merge_index_group_list))

    for pair in merge_index_pair_list:

        # check if either value in pair is already registered in a group

        added_to_group_flag = False
        for group in merge_index_group_list:
            if pair[0] in group or pair[1] in group:
                group.append(pair[0])
                group.append(pair[1])
                added_to_group_flag = True
                break

        if not added_to_group_flag:
            # if no group has the index values in pair, then create a new group

            new_group = list(pair)
            merge_index_group_list.append(new_group)

    # modify each group to have unique values only once

    modified_merge_index_group_list = []

    for group in merge_index_group_list:
        unique_list = list(set(group))
        modified_merge_index_group_list.append(unique_list)

    merge_index_group_list = modified_merge_index_group_list

    dl.debug("merge_index_group_list after unique filtered: {}".format(merge_index_group_list))

    # first filter out the line infos that are not subject to any merges

    all_index_subject_to_merges = []

    for group in merge_index_group_list:
        all_index_subject_to_merges += group

    after_merge_line_info_list = []

    # first populate the lineinfo that is not subject to any kind of merge
    for index in range(len(line_info_list)):
        if index not in all_index_subject_to_merges:
            after_merge_line_info_list.append(line_info_list[index])

    # apply merge to each groups

    merged_line_info_list = []

    for group in merge_index_group_list:
        group_lineinfo_list = []

        for index in group:
            group_lineinfo_list.append(line_info_list[index])

        extended_lineinfo = _merge_by_extending_longest_line(group_lineinfo_list)
        merged_line_info_list.append(extended_lineinfo)

    # append the merged lineinfos to the output list

    after_merge_line_info_list.extend(merged_line_info_list)

    return after_merge_line_info_list


def _merge_by_extending_longest_line(line_info_list):
    if len(line_info_list) == 1:
        return line_info_list[0]

    # find the longest line
    length_list = []
    min_x1 = None
    max_x2 = None
    min_y1 = None
    max_y2 = None

    for lineinfo in line_info_list:

        length_list.append(_get_length_of_lineinfo(lineinfo))

        if min_x1 is None or lineinfo[0] < min_x1:
            min_x1 = lineinfo[0]

        if max_x2 is None or lineinfo[2] > max_x2:
            max_x2 = lineinfo[2]

        if min_y1 is None or lineinfo[1] < min_y1:
            min_y1 = lineinfo[1]

        if max_y2 is None or lineinfo[3] > max_y2:
            max_y2 = lineinfo[3]

    length_nparry = np.array(length_list)
    longest_index = np.argmax(length_nparry)
    longest_lineinfo = line_info_list[longest_index]

    # base on the longest one, extend it to the min/max xy values

    slope = _get_slope_of_lineinfo(longest_lineinfo)

    # extend from x1,y1

    x1 = longest_lineinfo[0]
    y1 = longest_lineinfo[1]

    ext_x1, ext_y1 = _extend_from_point_with_slope_to_target_point(longest_lineinfo[:2], slope, (min_x1, min_y1))
    ext_x2, ext_y2 = _extend_from_point_with_slope_to_target_point(longest_lineinfo[2:], slope, (max_x2, max_y2))

    dl.debug("ext values: {} {} {} {}".format(ext_x1, ext_y1, ext_x2, ext_y2))

    # ensure int type
    ext_x1 = int(ext_x1)
    ext_y1 = int(ext_y1)
    ext_x2 = int(ext_x2)
    ext_y2 = int(ext_y2)

    extended_lineinfo = [ext_x1, ext_y1, ext_x2, ext_y2]

    return extended_lineinfo


def clm_vertical_lines_v1(line_info_list, slope_diff_threshold=0.5, gap_percentage_threshold=0.1,
                          return_debug_bundle=True):
    """
    do continuous line merge with given line infos and return the merged results

    """

    # calculate slopes of all line_info_list

    inverse_slope_list = []

    for lineinfo in line_info_list:
        slope = _get_inverse_slope_of_lineinfo(lineinfo)
        inverse_slope_list.append(slope)

    merge_index_pair_list = []

    for i in range(len(line_info_list)):
        for j in range(i + 1, len(line_info_list)):

            dl.debug("i={} , j={}".format(i, j))

            line_info_1 = line_info_list[i]
            line_info_2 = line_info_list[j]

            # if the lines are not "non-inclusive overlap" mode, then there is no possibility of merge

            if not _check_no_overlap_vertically(line_info_1, line_info_2):

                if not _check_non_inclusive_overlap_vertically(line_info_1, line_info_2):
                    dl.debug("not non_inclusive_overlap_horizontally")
                    continue
                else:
                    dl.debug("non inclusive overlap horizontally pass!")

            slope_1 = inverse_slope_list[i]
            slope_2 = inverse_slope_list[j]

            # evaluate is slope 1 and 2 are similar (within threshold)

            slope_diff = abs(slope_2 - slope_1)

            if slope_diff > slope_diff_threshold:
                continue

            # evaluate if the gap is small
            gap_length = _get_gap_length(line_info_1, line_info_2)
            gap_length_threshold = gap_percentage_threshold * min(_get_length_of_lineinfo(line_info_1),
                                                                  _get_length_of_lineinfo(line_info_2))
            dl.debug("gap_length={} , gap_length_threshold: {}".format(gap_length, gap_length_threshold))

            if gap_length > gap_length_threshold:
                dl.debug("gap length too large")
                continue
            else:
                dl.debug("gap length pass!")

            pair = (i, j)
            merge_index_pair_list.append(pair)

    dl.debug("merge_index_pair_list: {}".format(merge_index_pair_list))

    ## do merge

    # first union the pairs into groups

    merge_index_group_list = []

    dl.debug("merge_index_group_list: {}".format(merge_index_group_list))

    for pair in merge_index_pair_list:

        # check if either value in pair is already registered in a group

        added_to_group_flag = False
        for group in merge_index_group_list:
            if pair[0] in group or pair[1] in group:
                group.append(pair[0])
                group.append(pair[1])
                added_to_group_flag = True
                break

        if not added_to_group_flag:
            # if no group has the index values in pair, then create a new group

            new_group = list(pair)
            merge_index_group_list.append(new_group)

    # modify each group to have unique values only once

    modified_merge_index_group_list = []

    for group in merge_index_group_list:
        unique_list = list(set(group))
        modified_merge_index_group_list.append(unique_list)

    merge_index_group_list = modified_merge_index_group_list

    dl.debug("merge_index_group_list after unique filtered: {}".format(merge_index_group_list))

    # first filter out the line infos that are not subject to any merges

    all_index_subject_to_merges = []

    for group in merge_index_group_list:
        all_index_subject_to_merges += group

    after_merge_line_info_list = []

    # first populate the lineinfo that is not subject to any kind of merge
    for index in range(len(line_info_list)):
        if index not in all_index_subject_to_merges:
            after_merge_line_info_list.append(line_info_list[index])

    # apply merge to each groups

    merged_line_info_list = []
    merge_group_list = []

    for group in merge_index_group_list:
        group_lineinfo_list = []

        for index in group:
            group_lineinfo_list.append(line_info_list[index])

        merge_group_list.append(group_lineinfo_list)

        extended_lineinfo = _merge_by_extending_longest_line(group_lineinfo_list)
        merged_line_info_list.append(extended_lineinfo)

    # append the merged lineinfos to the output list

    after_merge_line_info_list.extend(merged_line_info_list)

    # prepare debug bundle

    debug_bundle = {
        "merge_group_list": merge_group_list
    }

    if return_debug_bundle:
        return after_merge_line_info_list, debug_bundle
    else:
        return after_merge_line_info_list


def _merge_by_extending_longest_line(line_info_list):
    if len(line_info_list) == 1:
        return line_info_list[0]

    # find the longest line
    length_list = []
    min_x1 = None
    max_x2 = None
    min_y1 = None
    max_y2 = None

    for lineinfo in line_info_list:

        length_list.append(_get_length_of_lineinfo(lineinfo))

        if min_x1 is None or lineinfo[0] < min_x1:
            min_x1 = lineinfo[0]

        if max_x2 is None or lineinfo[2] > max_x2:
            max_x2 = lineinfo[2]

        if min_y1 is None or lineinfo[1] < min_y1:
            min_y1 = lineinfo[1]

        if max_y2 is None or lineinfo[3] > max_y2:
            max_y2 = lineinfo[3]

    length_nparry = np.array(length_list)
    longest_index = np.argmax(length_nparry)
    longest_lineinfo = line_info_list[longest_index]

    # base on the longest one, extend it to the min/max xy values

    slope = _get_slope_of_lineinfo(longest_lineinfo)

    # extend from x1,y1

    x1 = longest_lineinfo[0]
    y1 = longest_lineinfo[1]

    ext_x1, ext_y1 = _extend_from_point_with_slope_to_target_point(longest_lineinfo[:2], slope, (min_x1, min_y1))
    ext_x2, ext_y2 = _extend_from_point_with_slope_to_target_point(longest_lineinfo[2:], slope, (max_x2, max_y2))

    dl.debug("ext values: {} {} {} {}".format(ext_x1, ext_y1, ext_x2, ext_y2))

    # ensure int type
    ext_x1 = int(ext_x1)
    ext_y1 = int(ext_y1)
    ext_x2 = int(ext_x2)
    ext_y2 = int(ext_y2)

    extended_lineinfo = [ext_x1, ext_y1, ext_x2, ext_y2]

    return extended_lineinfo


def _extend_from_point_with_slope_to_target_point(starting_point, slope, limit_point):
    """
    extend from starting point with slop toward as much as possible within the boundaries limited by limit point. 
    return the extrapolated point (x,y)

    :param starting_point: tuple or list length 2. (x1,y1)
    :param slope: dy/dx slope float value
    :param limit_point: tuple or list with length 2. (x1,y1)
    """

    xs = starting_point[0]
    ys = starting_point[1]

    xl = limit_point[0]
    yl = limit_point[1]

    dl.debug("starting point: {} , limit_point: {}".format(starting_point, limit_point))

    delta_y_limit = yl - ys
    delta_x_limit = xl - xs

    dl.debug("delta_y_limit: {} , delta_x_limit: {}".format(delta_y_limit, delta_x_limit))

    delta_y_from_delta_x = delta_x_limit * slope

    dl.debug("delta_y_from_delta_x: {}".format(delta_y_from_delta_x))

    if delta_y_limit > 0:

        if delta_y_from_delta_x > delta_y_limit:
            delta_y_det = delta_y_limit
        else:
            delta_y_det = max(0, delta_y_from_delta_x)
    else:
        if delta_y_from_delta_x < delta_y_limit:
            delta_y_det = delta_y_limit
        else:
            delta_y_det = min(0, delta_y_from_delta_x)

    ## account for the case when delta_x_limit is 0 but delta_y_limit is nonzero
    if delta_x_limit == 0:
        if slope > 0:
            delta_y_det = delta_y_limit
        else:
            delta_y_det = delta_y_limit

    if slope == 0:
        delta_x_det = delta_x_limit
    else:
        delta_x_det = delta_y_det / slope

    dl.debug("delta_x_det: {} , delta_y_det: {}, slope={}".format(delta_x_det, delta_y_det, slope))

    extrapolate_point_x = xs + delta_x_det
    extrapolate_point_y = ys + delta_y_det

    return (extrapolate_point_x, extrapolate_point_y)
