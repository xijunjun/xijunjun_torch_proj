#coding:utf-8
import  numpy as np
import  cv2
import os,sys
import platform
SCREEN_WIDTH=1800                    #图像窗口最大宽度
SCREEN_HEIGHT=900                    #图像窗口最大高度
local_path = u"/media/tao/project/workspace/plate2018/data/车牌标注/imgdata/标注完成/"
imgpathlist=os.listdir(local_path)

def limit_window(disimg,winnane):
    wm_ratio=1.0
    if disimg.shape[1] > SCREEN_WIDTH or disimg.shape[0] > SCREEN_HEIGHT:
        if (disimg.shape[1] / float(disimg.shape[0])) > (SCREEN_WIDTH / float(SCREEN_HEIGHT)):
            cv2.resizeWindow(winnane, SCREEN_WIDTH, int(SCREEN_WIDTH / float(disimg.shape[1]) * disimg.shape[0]))
            wm_ratio = SCREEN_WIDTH / float(disimg.shape[1])
        else:
            cv2.resizeWindow(winnane, int(SCREEN_HEIGHT / float(disimg.shape[0]) * disimg.shape[1]), SCREEN_HEIGHT)
            wm_ratio = SCREEN_HEIGHT / float(disimg.shape[0])
    else:
        cv2.resizeWindow(winnane, disimg.shape[1], disimg.shape[0])
    return wm_ratio
def file_extension(path):
  return os.path.splitext(path)[1]
def _load_pascal_annotation(_data_path):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    lines=open(_data_path).readlines()
    num_objs=len(lines)
    boxes = np.zeros((num_objs, 8), dtype=np.uint16)
    charbdrcts=[]
    for i,oneline in enumerate(lines):
        charrcts=[]
        item= lines[i].decode('utf-8').rstrip().split(',')
        boxes[i, :] = [int(num) for num in item[0].split(' ')[0:8]]
        platestr=item[2]
        print len(platestr)
        for j in xrange(len(platestr)):
            charrcts.append([int(num) for num in item[3+j].split(' ')[0:8]])
        charbdrcts.append(charrcts)
    charbdrcts=np.array(charbdrcts)
    # print charbdrcts
    return {'boxes': boxes,
            'charbdrcts':charbdrcts
            }
def encode_thr_sys(tstr):
    return tstr.encode('gbk') if 'Windows' in platform.system() else tstr.encode('utf-8')
if __name__ == '__main__':
    for i_img in imgpathlist:
        if  file_extension(i_img) not in ['.JPG','.bmp','.PNG','.png','.jpeg','.jpg']:
            continue
        impath=local_path+i_img
        picano=_load_pascal_annotation(impath.split('.')[0]+'.txt')
        ori_img = cv2.imread(encode_thr_sys(local_path + i_img))

        boxes=picano['boxes']
        charbdrcts=picano['charbdrcts']
        for i in range(boxes.shape[0]):
            for j in range(4):
                cv2.line(ori_img, (boxes[i][j*2], boxes[i][j*2+1]), (boxes[i][(j+1)%4*2], boxes[i][(j+1)%4*2+1]), (0, 0, 255), thickness=2)
            for charboxes in charbdrcts[i]:
                for j in range(4):
                    cv2.line(ori_img, (charboxes[j * 2], charboxes[j * 2 + 1]),
                             (charboxes[(j + 1) % 4 * 2], charboxes[(j + 1) % 4 * 2 + 1]), (0, 0, 255), thickness=1)
        cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
        limit_window( ori_img,'img')
        cv2.imshow('img',ori_img)

        cv2.waitKey()
