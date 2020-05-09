import cv2, os, shutil
from table_border_extraction_fns import get_h_and_v_line_bbox, clTable, get_h_and_v_line_bbox_CNX, filter_lines_error, detect_table


testimg_path = "C:/Users\chungnx/Desktop/aicr_data_hw/image_crop/aicrhw_2020-02-26_16-54/AICR_P000016/origine.jpg"

output_dir = "outputs/table_template"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

img = cv2.imread(testimg_path)
print(type(img))

hline_list, vline_list = get_h_and_v_line_bbox_CNX(img)
# sorted(hline_list)
list_p_table, hline_list, vline_list = detect_table(hline_list, vline_list)
# list_p_table.sort(key = lambda  x:x.startY)

# draw lines


copyimg = img.copy()
overlay = img.copy()
count = 1
countb = 1
for l_p in list_p_table:
    font = cv2.FONT_HERSHEY_SIMPLEX
    l_p.detect_cells()
    cv2.circle(img,(l_p.startX,l_p.startY),7,(count*50,count*85,count*40),-1)
    cv2.putText(img, "tb "+str(countb), (l_p.startX, l_p.startY), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    countb += 1
    cv2.circle(img, (l_p.endX, l_p.endY), 7, (count*50,count*85, count*40),-1)
    cv2.rectangle(img,(l_p.startX, l_p.startY),(l_p.endX, l_p.endY), (count*50,count*85,count*40),1)
    for v in l_p.listVertical:
        cv2.line(img, (v[0], v[1]), (v[2], v[3]), (50*count,count*70,count*30), 1)
    for h in l_p.listHorizontal:
        cv2.line(img, (h[0], h[1]), (h[2], h[3]), (50*count, count *70, count*30), 1)
    for p in l_p.listPoints:
        cv2.circle(img, (p[0], p[1]), 7, (count * 50, count * 85, count * 40), -1)
    count_cell = 0
    print("len cell ", len(l_p.listCells))
    for cell in l_p.listCells:
        count_cell += 1
        centerx = int((cell[2] - cell[0]) / 2 + cell[0])
        centery = int((cell[3] - cell[1]) / 2 + cell[1])
        cv2.putText(img, str(count_cell), (centerx, centery), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.rectangle(img,(cell[0],cell[1]),(cell[2],cell[3]),
        #               ((count_cell + 5) * 6, (count_cell + 5) * 5, count_cell * 1),1)
    count+=1
# img = cv2.addWeighted(overlay, 0.7, img, 1 - 0.7, 0)
savepath = os.path.join(output_dir, "hvline3.png")
cv2.imshow("table", img)
cv2.waitKey()
