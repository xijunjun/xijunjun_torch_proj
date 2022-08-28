#coding:utf-8
# --------------------------------------------------------
# KB537_TEXT_GROUP
# 2018-3-26
# --------------------------------------------------------
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import os
import numpy as np
import shutil
import platform
from  gen_plate_std  import plate_temp_list,get_stand_plate,char_dict

############路径及窗口大小设置############################
local_path = u"F:/workspace/plate2018/data/车牌标注/imgdata"          #图像根目录
MAX_WIDTH=1800                    #图像窗口最大宽度
MAX_HEIGHT=1000                    #图像窗口最大高度
########################################



local_path=local_path+'/'
key_dic={}
def load_key_val():
    key_val_path='key_val.txt'
    if 'Windows' in platform.system():
        key_val_path='key_val_win.txt'
    lines=open(key_val_path).readlines()
    for line in lines:
        item=line.split(' ')
        vals=item[1].split(',')
        val_lst=[]
        for val in vals:
            val_lst.append(int(val))
        key_dic[item[0]]=val_lst
        # print item[0],val_lst
load_key_val()

imgpathlist=os.listdir(local_path)
only4points=0
plate_anno,ptlist,platelist,global_var,charbdrdcts=[],[],[],[],[]

def file_extension(path):
  return os.path.splitext(path)[1]
def sum_img_vertical(imglist,inter):
    rows=0;cols=0
    for img in imglist:
        rows+=img.shape[0]+inter
        if img.shape[1]>cols:
            cols=img.shape[1]
    sumimg = np.zeros((rows-inter, cols, 3), np.uint8)
    ystart = 0
    for img in imglist:
        sumimg[ystart:ystart+img.shape[0],0:img.shape[1]]=img
        ystart=ystart+img.shape[0]+inter
    return sumimg
def sum_img_hori(imglist,inter):
    rows=0;cols=0
    for img in imglist:
        cols+=img.shape[1]+inter
        if img.shape[0]>rows:
            rows=img.shape[0]
    sumimg = np.zeros((rows, cols-inter,3), np.uint8)
    xstart = 0
    for img in imglist:
        sumimg[0:img.shape[0],xstart:xstart+img.shape[1]]=img
        xstart=xstart+img.shape[1]+inter
    return sumimg
def limit_imgw(img):
    if  img.shape[1]>360:
        img=cv2.resize( img,(360,int(360.0/img.shape[1]*img.shape[0])),interpolation=cv2.INTER_CUBIC )
    if  img.shape[0]>360:
        img=cv2.resize( img,(int(360.0/img.shape[0]*img.shape[1]),360),interpolation=cv2.INTER_CUBIC )
    return  img
def refresh_ori():
    disimg=ori_img.copy()
    for pt in ptlist:
        cv2.circle(disimg, (pt[0], pt[1]),int(4/ global_var[0]), (0, 0, 255), thickness=-1)
    for i in range(0,len(ptlist)):
        if (i+1)%4>=len(ptlist):break
        cv2.line(disimg,tuple(ptlist[i%4]),tuple(ptlist[(i+1)%4]), (0, 0,255),int(2/ global_var[0]))
    for oneplate in platelist:
        for i in range(0,4):
            cv2.line(disimg, tuple(oneplate[i % 4]), tuple(oneplate[(i + 1) % 4]), (0, 255, 0), int(2 / global_var[0]))
    cv2.imshow('img', disimg)
###################################
def get_bfrect(spt):
    tlx = min([spt[0], spt[2], spt[4], spt[6]])
    tly = min([spt[1], spt[3], spt[5], spt[7]])
    brx = max([spt[0], spt[2], spt[4], spt[6]])
    bry = max([spt[1], spt[3], spt[5], spt[7]])

    return [tlx,tly,brx,bry]
def get_cont(img, fourpts):
    bin_img = img.copy()
    for j in range(4):
        cv2.line(bin_img, (fourpts[j * 2], fourpts[j * 2 + 1]), (fourpts[(j + 1) % 4 * 2],
                                                                 fourpts[(j + 1) % 4 * 2 + 1]), (255), thickness=1)
    myimg, contours, h = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
#####################################
def equalize_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
    # hsv[:, :, 2] = hsv[:, :, 2] * (basev + np.random.random() * (1 - basev));
    # hsv[:, :, 1] = cv2.equalizeHist(hsv[:, :, 1])
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2] )
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR);
    return img
def refresh_preview(tpimg,offpt):
    global goodimg
    tempdict = plate_temp_list[global_var[6]]
    disimg=tpimg.copy()
    disimg=equalize_hsv(disimg)

    bdrct = cv2.boundingRect(np.array(ptlist))
    minbdrct = cv2.minAreaRect(np.array(ptlist))
    height =float(min(minbdrct[1][0], minbdrct[1][1]))
    width = height/tempdict['height']*tempdict['width']
    height,width=int(height),int(width)

    oripts = [[ptlist[0][0]-bdrct[0], ptlist[0][1]-bdrct[1]], [ptlist[1][0]-bdrct[0], ptlist[1][1]-bdrct[1]],
              [ptlist[2][0]-bdrct[0], ptlist[2][1]-bdrct[1]], [ptlist[3][0]-bdrct[0], ptlist[3][1]-bdrct[1]] ]

    srcpts = [[ptlist[0][0]-offpt[0], ptlist[0][1]-offpt[1]], [ptlist[1][0]-offpt[0], ptlist[1][1]-offpt[1]],
              [ptlist[2][0]-offpt[0], ptlist[2][1]-offpt[1]], [ptlist[3][0]-offpt[0], ptlist[3][1]-offpt[1]] ]

    rctpts=[[0,0], [width,0], [width,height], [0,height]]
    pt_tl=[bdrct[0],bdrct[1]]
    pt_rd=[bdrct[0]+ bdrct[2]-1,bdrct[1]+ bdrct[3]-1]
    goodimg = ori_img[pt_tl[1]:pt_rd[1] + 1, pt_tl[0]:pt_rd[0] + 1].copy()
    M=cv2.getPerspectiveTransform(np.array(oripts,dtype="float32"),np.array(rctpts,dtype="float32"))
    goodimg=cv2.warpPerspective(goodimg,M,(width,height))
    goodimg = equalize_hsv(goodimg)
    disimg=sum_img_vertical([disimg,goodimg],10)
    #limit_window(disimg, 'preview')
    wm_ratio = get_ratio(disimg)

    linecolor=(255-tempdict['bkcolor'][0],255-tempdict['bkcolor'][1],255-tempdict['bkcolor'][2])
    for i in range(0,4):
        cv2.line(disimg,(ptlist[i%4][0]-offpt[0],ptlist[i%4][1]-offpt[1]),(ptlist[(i+1)%4][0]-offpt[0],
                    ptlist[(i+1)%4][1]-offpt[1]),linecolor,int(2/wm_ratio))
    cv2.circle(disimg, (ptlist[global_var[1]][0] - offpt[0],ptlist[global_var[1]][1] - offpt[1]), int(6 / wm_ratio), (0, 255, 0), thickness=-1)
    for pt in ptlist:
        cv2.circle(disimg, (pt[0]-offpt[0], pt[1]-offpt[1]),int(4/ wm_ratio), (0, 0, 255), thickness=-1)


    tempw = tempdict['width']
    temph = tempdict['height']
    src_pts = [[0, 0], [tempw - 1, 0], [tempw - 1, temph - 1], [0, temph - 1]]
    # temprct=tempdict['bdrct']
    # tempw = temprct[2]-temprct[0]+1
    # temph = temprct[3]-temprct[1]+1
    # src_pts = [[temprct[0], temprct[1]], [temprct[2], temprct[1]], [temprct[2], temprct[3]], [temprct[0], temprct[3]]]
    M = cv2.getPerspectiveTransform(np.array(src_pts, dtype="float32"), np.array(srcpts, dtype="float32"))
    charrct = tempdict['char_rct']
    allboxes = []
    del charbdrdcts[:]
    for rct in charrct:
        new_4pts = []
        absolute_4pts = []
        fourpts = [[rct[0], rct[1]], [rct[2], rct[1]],
                   [rct[2], rct[3]], [rct[0], rct[3]]]
        for pt in fourpts:
            new_cent = np.array([[pt[0]], [pt[1]], [1]])
            new_cent = np.dot(M, new_cent)
            # print new_cent
            new_cent[0][0] /= new_cent[2][0]
            new_cent[1][0] /= new_cent[2][0]
            new_4pts.append([int(new_cent[0][0]), int(new_cent[1][0])])
            absolute_4pts.append([int(new_cent[0][0])+offpt[0], int(new_cent[1][0])+offpt[1]])
            # new_4pts.append([pt[0], pt[1]])
        allboxes.append(np.array(new_4pts).copy())
        charbdrdcts.append(np.array(absolute_4pts).copy())

    bin_img = np.zeros((disimg.shape[0], disimg.shape[1], 1), disimg.dtype)
    mask_img = np.zeros((disimg.shape[0], disimg.shape[1], 3), disimg.dtype)


    cutcharlist=[]

    for j in range(tempdict['numchar']):
        fourpts = [allboxes[j][0][0], allboxes[j][0][1], allboxes[j][1][0], allboxes[j][1][1],
                   allboxes[j][2][0], allboxes[j][2][1], allboxes[j][3][0], allboxes[j][3][1]]
        # fourpts = dilate_pts(fourpts, -0.1, 2000, 2000)
        srcpts= [[allboxes[j][0][0], allboxes[j][0][1]], [allboxes[j][1][0], allboxes[j][1][1]],
                   [allboxes[j][2][0], allboxes[j][2][1]], [allboxes[j][3][0], allboxes[j][3][1]] ]

        # charbdrct = cv2.minAreaRect(np.array(srcpts))
        # width= min(charbdrct[1][0], charbdrct[1][1])
        # height=2*width
        # width= int(0.5*(charrct[j][2]-charrct[j][0]))
        # height=int(0.5*(charrct[j][3]-charrct[j][1]))
        width= 22
        height=44
        dstpts=[[0,0],[width,0],[width,height],[0,height]]
        M = cv2.getPerspectiveTransform(np.array(srcpts, dtype="float32"), np.array(dstpts, dtype="float32"))
        charimg = cv2.warpPerspective(tpimg.copy(), M, (int(width), int(height)))
        # cv2.imshow('11',charimg)
        # cv2.waitKey(0)
        cutcharlist.append(charimg.copy())
        contours = get_cont(bin_img, fourpts)
        cv2.drawContours(mask_img, contours, -1, tempdict['maskcolor'], thickness=-1)

    cutcharsumimg = sum_img_hori(cutcharlist,5)
    cutcharsumimg =cv2.resize(cutcharsumimg,(goodimg.shape[1],int(float(goodimg.shape[1])/cutcharsumimg.shape[1]*cutcharsumimg.shape[0])))
    # stdplates = cv2.resize(stdplates, (int(stdplates.shape[1] * resizeratio), int(stdplates.shape[0] * resizeratio)),
    #                        stdplates)

    mask_img=cv2.cvtColor(mask_img,cv2.COLOR_BGR2BGRA)
    disimg = cv2.cvtColor(disimg, cv2.COLOR_BGR2BGRA)

    graydisimg = cv2.cvtColor(goodimg, cv2.COLOR_BGR2GRAY)
    avggray=np.sum(graydisimg)/(graydisimg.shape[0]*graydisimg.shape[1])/255.0
    # print avggray

    disimg=cv2.addWeighted(disimg, 0.5+0.5*(1-avggray*1.5), mask_img, 0.5+0.5*avggray*1.5, 0)
    disimg = cv2.cvtColor(disimg, cv2.COLOR_BGRA2BGR)
    dis_stdplates=stdplates.copy()
    tl,br=stdplates_rct_dict[global_var[6]][0],stdplates_rct_dict[global_var[6]][1]
    cv2.rectangle(dis_stdplates,(tl[0],tl[1]),(br[0],br[1]), (0, 0, 255), 4)
    disimg = sum_img_vertical([disimg, cutcharsumimg],5)
    disimg=sum_img_hori([disimg,dis_stdplates],10)


    # cv2.setMouseCallback('preview', select_platetype)
    if disimg.shape[0]<350 or disimg.shape[1]<350:
        cv2.resizeWindow('preview', disimg.shape[1]*2, disimg.shape[0]*2)
    else:
        limit_window_wh(disimg, 'preview', 1200, 800)
    cv2.imshow('preview', disimg)
    # limit_window_wh(disimg, 'preview', 1200, 800)
def get4pts():
    cv2.namedWindow('preview', cv2.WINDOW_FREERATIO)
    cv2.moveWindow('preview',0,0)
    bdrct= cv2.boundingRect(np.array(ptlist))
    delta_w=int(bdrct[2]*0.3)
    delta_h=int(bdrct[3]*0.5)
    pt_tl=[bdrct[0],bdrct[1]]
    pt_rd=[bdrct[0]+ bdrct[2]-1,bdrct[1]+ bdrct[3]-1]
    pt_tl[0]=0 if pt_tl[0]-delta_w<0 else pt_tl[0]-delta_w
    pt_rd[0] =ori_img.shape[1]-1 if  pt_rd[0]+delta_w>ori_img.shape[1]-1 else pt_rd[0]+delta_w
    pt_tl[1] =0 if pt_tl[1]-delta_h<0 else pt_tl[1]-delta_h
    pt_rd[1] = ori_img.shape[0]-1 if pt_rd[1]+delta_h>ori_img.shape[0]-1 else pt_rd[1]+delta_h
    pltimg = ori_img[pt_tl[1]:pt_rd[1]+1, pt_tl[0]:pt_rd[0]+1].copy()
    refresh_preview(pltimg, pt_tl)
    while 1:
        key=cv2.waitKey(0)
        if key in key_dic['ENTER']:
            break
        if key in key_dic['SPACE']:
            global_var[1]=(global_var[1]+1)%4
        if key in key_dic['UP']:
            if ptlist[global_var[1]][1]-pt_tl[1]-global_var[2]>=0:
                ptlist[global_var[1]][1] -=global_var[2]
        if key in key_dic['DOWN']:
            if ptlist[global_var[1]][1]-pt_tl[1]+global_var[2]<=pltimg.shape[0]-1:
                ptlist[global_var[1]][1]  +=global_var[2]
        if key in key_dic['LEFT']:
            if ptlist[global_var[1]][0]-pt_tl[0]-global_var[2]>=0:
                ptlist[global_var[1]][0]  -=global_var[2]
        if key in key_dic['RIGHT']:
            if ptlist[global_var[1]][0]-pt_tl[0]+global_var[2]<=pltimg.shape[1]-1:
                ptlist[global_var[1]][0]  +=global_var[2]
        if key in key_dic['PLUS']:
            global_var[6]+=1
            global_var[6]%=len(stdplates_rct_dict)
        if key in key_dic['MINUS']:
            global_var[6] -= 1
            if global_var[6]<0:
                global_var[6] = len(stdplates_rct_dict)-1
        if key in key_dic['ESC']:
            del ptlist[:]
            # del charbdrdcts[:]
            cv2.destroyWindow('preview')
            return
        refresh_preview(pltimg, pt_tl)
        # refresh_ori()
    # print ptlist
    platelist.append(list(ptlist))#list.append([])只拷贝索引,不拷贝对象
    refresh_ori()
    cv2.destroyWindow('preview')
def draw_circle(event,x,y,flags,param):
    if len(ptlist) == 4:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        ptlist.append([x,y])
        refresh_ori()
        if len(ptlist)==4:
            get4pts()
            get_info()
            refresh_ori()
            cv2.waitKey(1)#注意此处等待按键

def limit_window(disimg,winnane):
    wm_ratio=1.0
    if disimg.shape[1] > MAX_WIDTH or disimg.shape[0] > MAX_HEIGHT:
        if (disimg.shape[1] / float(disimg.shape[0])) > (MAX_WIDTH / float(MAX_HEIGHT)):
            cv2.resizeWindow(winnane, MAX_WIDTH, int(MAX_WIDTH / float(disimg.shape[1]) * disimg.shape[0]))
            wm_ratio = MAX_WIDTH / float(disimg.shape[1])
        else:
            cv2.resizeWindow(winnane, int(MAX_HEIGHT / float(disimg.shape[0]) * disimg.shape[1]), MAX_HEIGHT)
            wm_ratio = MAX_HEIGHT / float(disimg.shape[0])
    else:
        cv2.resizeWindow(winnane, disimg.shape[1], disimg.shape[0])
    return wm_ratio
def limit_window_wh(disimg,winnane,newWIDTH,newHEIGHT):
    if disimg.shape[1] > newWIDTH or disimg.shape[0] > newHEIGHT:
        if (disimg.shape[1] / float(disimg.shape[0])) > (newWIDTH / float(newHEIGHT)):
            cv2.resizeWindow(winnane, newWIDTH, int(newWIDTH / float(disimg.shape[1]) * disimg.shape[0]))
        else:
            cv2.resizeWindow(winnane, int(newHEIGHT / float(disimg.shape[0]) * disimg.shape[1]), newHEIGHT)
    else:
        cv2.resizeWindow(winnane, disimg.shape[1], disimg.shape[0])
def get_ratio(disimg):
    wm_ratio=1.0
    if disimg.shape[1] > MAX_WIDTH or disimg.shape[0] > MAX_HEIGHT:
        if (disimg.shape[1] / float(disimg.shape[0])) > (MAX_WIDTH / float(MAX_HEIGHT)):
            wm_ratio = MAX_WIDTH / float(disimg.shape[1])
        else:
            wm_ratio = MAX_HEIGHT / float(disimg.shape[0])
    return wm_ratio
def make_dir(local_path):
    if os.path.isdir(local_path + u'标注完成') is False:
        os.mkdir(local_path + u'标注完成')
    if os.path.isdir(local_path + u'不合格') is False:
        os.mkdir(local_path + u'不合格')


def select_info(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        idx,idy=xy2id(x, y)
        if list((idx, idy))  in chi_dic.values():
            global_var[5] = get_dic_key_by_val(chi_dic, list((idx, idy)), global_var[5])
            if len(platestr) < 10:
                 platestr.append(global_var[5])
        refresh_dialog(newdialogimg)
def xy2id(x,y):
    return x/30,y/40
def dis_index_rec(srcimg,xy):
    x,y=xy[0],xy[1]
    cv2.rectangle(srcimg, (30 *x, 40 * y),(30 * (x+1), 40 * (y+1)), (0, 0, 255), 1)
def refresh_dialog(tpimg):
    disimg=tpimg.copy()
    dis_index_rec(disimg, chi_dic[global_var[5]])
    dis_platestr(disimg)
    cv2.imshow('dialog', disimg)
    cv2.moveWindow('dialog', 0, 0)
def gen_char(f,val,color):
    bkcolor,charcolor=color[0],color[1]
    img=Image.new("RGB", (30,40),bkcolor)
    draw = ImageDraw.Draw(img)
    draw.text((0, 6),val,charcolor,font=f)
    A = np.array(img)
    return A
def paste_img(src,x,y,val,color):
    img=gen_char(gfont, val,color)
    src[y:y+40,x:x+30,:]=img.copy()
def dis_platestr(img):
    displatestr=''
    for pstr in platestr:
        displatestr += pstr
    for i in range(len(platestr),plate_temp['numchar']):
        displatestr += '-'
    tpimg, rct = get_stand_plate(plate_temp_list[global_var[6]], displatestr)
    tpimg = cv2.resize(tpimg, (int(tpimg.shape[1] * 0.3), int(tpimg.shape[0] * 0.3)),tpimg)
    img[5*40:5*40+tpimg.shape[0],0:tpimg.shape[1]]=tpimg

def get_info():
    global newdialogimg
    if len(ptlist)==0:
        return
    if only4points==1:
        plate_anno.append('')
        del platestr[:]
        del ptlist[:]#重新初始化参数
        global_var[1] = 0
        return
    newdialogimg=sum_img_vertical([dialogimg,limit_imgw(goodimg)],10)
    refresh_dialog(newdialogimg)
    cv2.setMouseCallback('dialog', select_info)
    # cv2.imshow('goodimg',goodimg)
    while 1:
        key=cv2.waitKey(0)
        if key in key_dic['ENTER']:
            anno_str=str(global_var[6])+','
            for pchar in platestr:
                anno_str+=pchar
            anno_str += ','
            print anno_str
            for item in charbdrdcts:
                anno_str +=str(item[0][0])+' '+str(item[0][1])+' '+str(item[1][0])+' '+str(item[1][1])+' '+str(item[2][0])+' '+str(item[2][1])+' '+str(item[3][0])+' '+str(item[3][1])+','
            anno_str=anno_str.rstrip(',')
            plate_anno.append(anno_str)
            del platestr[:]
            # global_var[3] = 'blu'
            break
        if key in key_dic['BACK']:
            if len(platestr)>0:
                platestr.pop()
        if ((key>=ord('0') and key<=ord('9')) or (key>=ord('a') and key<=ord('z')) or (key>=ord('A') and key<=ord('Z'))) and chr(key).upper() in engnumtable:
            if len(platestr) < plate_temp_list[global_var[6]]['numchar']:
                platestr.append(chr(key).upper())
        refresh_dialog(newdialogimg)
    cv2.destroyWindow('dialog')
    del ptlist[:]#重新初始化参数
    # del charbdrdcts[:]
    global_var[1] = 0
def get_dic_key_by_val(src_dic,dstval,rtval):
    if dstval not in src_dic.values():return rtval
    for key, value in src_dic.items():
        if value==dstval:
            return  key
def encode_thr_sys(tstr):
    return tstr.encode('gbk') if 'Windows' in platform.system() else tstr.encode('utf-8')

if __name__ == '__main__':
    stopflag=0
    platestr=[]
    global_var.append(1.0)#缩放尺度参数，画线和点的依据
    global_var.append(0)#cursor点选择
    global_var.append(1)#move_step调整点时的步长
    global_var.append(0)  #
    global_var.append(0)#
    global_var.append(u'京')  #
    global_var.append(0)#plate_type
    gfont = ImageFont.truetype("./font/simhei.ttf", 30, 0)

    chi_dic={}
    engnumtable='0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
    chi_table = u'京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼警港澳挂领使学字'
    stdplatelist=[]
    resizeratio=0.25
    stdplates_rct_dict = {}
    stdyoffset=0
    for i,plate_temp in enumerate(plate_temp_list):
        tpimg,rct=get_stand_plate(plate_temp, plate_temp['platestr'])
        stdplatelist.append(tpimg.copy())
        xcoord=int(resizeratio*(plate_temp['width'])-1)
        ycoord=int(resizeratio*(plate_temp['height'])-1)
        stdplates_rct_dict[i]=[[0,stdyoffset],[xcoord,stdyoffset+ycoord]]
        stdyoffset +=ycoord+1
    stdplates=sum_img_vertical(stdplatelist,0)
    stdplates=cv2.resize(stdplates,(int(stdplates.shape[1]*resizeratio),int(stdplates.shape[0]*resizeratio)),stdplates)


    for i,chi in enumerate(chi_table):
        chi_dic[chi]=[i%11,1+i/11]

    cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
    cv2.setMouseCallback('img', draw_circle)
    make_dir(local_path)
    dialogimg=cv2.imread(('dialog.jpg'))

    # cv2.imshow('11',stdplates)

    for i_img in imgpathlist:
        if  file_extension(i_img).lower() not in ['.bmp','.png','.jpeg','.jpg']:
            continue
        print local_path + i_img
        ori_img = cv2.imread(encode_thr_sys(local_path + i_img))
        global_var[0]=limit_window(ori_img, 'img')
        cv2.imshow('img',ori_img )
        cv2.moveWindow('img', 0, 0)
        while 1:
            key=cv2.waitKey(0)
            if key in key_dic['ESC']:
                stopflag=1
                break
            if key in key_dic['BACK']:
                if len(ptlist)>0:
                    ptlist.pop()
                elif len(plate_anno)>0:
                    plate_anno.pop()
                    platelist.pop()
            if key in key_dic['DELETE']:
                shutil.move((local_path + i_img),((local_path+u'不合格/')+ i_img))
                break
            if key in key_dic['ENTER']:
                if len(platelist)==0:
                    break
                lines=''
                for i,oneplate in enumerate(platelist):
                    lines += str(oneplate[0][0]) + ' ' + str(oneplate[0][1]) + ' ' + str(oneplate[1][0]) + ' ' + str(oneplate[1][1]) + ' ' + str(oneplate[2][0]) + ' ' +\
                             str(oneplate[2][1]) + ' ' + str(oneplate[3][0]) + ' ' + str(oneplate[3][1]) +','+plate_anno[i]+'\n'
                with open(local_path+u'标注完成/'+os.path.splitext(i_img)[0]+'.txt', 'w') as f:
                    f.write((lines.rstrip(' \n')).encode('utf-8'))
                shutil.move((local_path + i_img), (local_path + u'标注完成/' + i_img))
                break
            refresh_ori()
        if stopflag:
            break
        ptlist = []
        del platelist[:]
        del plate_anno[:]
        global_var[0]=1.0
        global_var[1] = 0

