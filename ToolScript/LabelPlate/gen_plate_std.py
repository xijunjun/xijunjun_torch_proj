# -*- coding: utf-8 -*-
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2;
import numpy as np;
import os;

ESC=27

plate1_temp={'name':'单行蓝牌','bkcolor':(155,0,0),'chcolor':(255,255,255),'width':440,'height':140,'numchar':7,
             'dotxyr':[134,70,5],'nailxyr':[[107,12,5],[332,12,5],[107,127,5],[332,127,5]],
            'sp_color':[],'bdrct':[15,25,196+57*4, 115],
            'char_rct':[[15,25,60,115],[15+57,25,60+57,115],[151,25,196,115], [151+57*1, 25, 196+57*1, 115],
                    [151+57*2, 25, 196+57*2, 115], [151+57*3, 25, 196+57*3, 115], [151+57*4, 25, 196+57*4, 115]],
             'platestr':u'京AF0236','maskcolor':(0,0,155)}

plate2_temp={'name':'单行黄牌','bkcolor':(0,200,250),'chcolor':(0,0,0),'width':440,'height':140,'numchar':7,
             'dotxyr':[134,70,5],'nailxyr':[[107,12,5],[332,12,5],[107,127,5],[332,127,5]],
            'sp_color':[],'bdrct':[15,25,196+57*4, 115],
            'char_rct':[[15,25,60,115],[15+57,25,60+57,115],[151,25,196,115], [151+57*1, 25, 196+57*1, 115],
                    [151+57*2, 25, 196+57*2, 115], [151+57*3, 25, 196+57*3, 115], [151+57*4, 25, 196+57*4, 115]],
             'platestr': u'京AF0236','maskcolor':(0,120,0)}

plate3_temp={'name':'双行黄牌','bkcolor':(0,200,250),'chcolor':(0,0,0),'width':440,'height':220,'numchar':7,
             'dotxyr':[220,47,5],'nailxyr':[[92,7,5],[338,7,5],[92,213,5],[338,213,5]],
            'sp_color':[],'bdrct':[27,15,92+80*4,200],
            'char_rct':[[107,15,187,75],[107+140,15,187+140,75],[27,90,92,200], [27+80*1,90,92+80*1,200],
                    [27+80*2,90,92+80*2,200], [27+80*3,90,92+80*3,200], [27+80*4,90,92+80*4,200]],
             'platestr': u'京AF0236','maskcolor':(0,125,0)}

plate4_temp={'name':'警车','bkcolor':(200,200,200),'chcolor':(0,0,0),'width':440,'height':140,'numchar':7,
             'dotxyr':[77,70,7],'nailxyr':[[107,12,5],[332,12,5],[107,127,5],[332,127,5]],
             'sp_color':[(7,(0,0,220))],'bdrct':[15,25,196+57*4, 115],
            'char_rct':[[15,25,60,115],[94,25,139,115],[94+57*1,25,139+57*1,115] ,[94+57*2,25,139+57*2,115],
                    [94+57*3,25,139+57*3,115], [94+57*4,25,139+57*4,115], [94+57*5,25,139+57*5,115]],
             'platestr': u'京AF023警','maskcolor':(0,0,255)}
plate5_temp={'name':'使馆车牌','bkcolor':(0,0,0),'chcolor':(255,255,255),'width':440,'height':140,'numchar':7,
             'dotxyr':[248,70,5],'nailxyr':[[107,12,5],[332,12,5],[107,127,5],[332,127,5]],
             'sp_color':[(1,(0,0,200))],'bdrct':[15,25,196+57*4, 115],
            'char_rct':[[15,25,60,115],[15+57*1,25,60+57*1,115],[15+57*2,25,60+57*2,115] ,[15+57*3,25,60+57*3,115],
                    [15+57*3+79,25,60+57*3+79,115], [15+57*4+79,25,60+57*4+79,115], [15+57*5+79,25,60+57*5+79,115]],
             'platestr': u'使AF0236','maskcolor':(0,0,255)}


plate_temp_list=[plate1_temp,plate2_temp,plate3_temp,plate4_temp,plate5_temp]
def file_extension(path):
  return os.path.splitext(path)[1]
def listdir_img(path):
    imglist=os.listdir(path)
    imglst=[]
    for m_img in imglist:
        if  file_extension(m_img) not in ['.JPG','.bmp','.PNG','.png','.jpeg','.jpg']:
            continue
        imglst.append(m_img)
    return imglst

def gen_char_dict(fc,fe):
    # imlst1,imlst2=[],[]
    chardict={}
    for val in engnumtable:
        # imlst1.append(gen_char(fe,val,45,90))
        f=fe
        img=gen_char(f,val,45,90)
        if val in [' ']:
            chardict[val] = 255-np.zeros((45,90,1))
        if val in ['A']:
            # f=fc
            kernel = np.uint8(np.zeros((5, 5)))
            for x in range(5):
                kernel[x, 2] = 1;
                kernel[2, x] = 1;
            # img = cv2.erode(img, kernel);
        chardict[val]=img
    for val in chi_table:
        # imlst2.append(gen_char(fc,val,45,90))
        chardict[val] = gen_char(fc, val, 45, 90)
    # return sum_img_vertical([sum_img_hori(imlst1),sum_img_hori(imlst2)])
    return chardict
def gen_char(f,val,wid,hei):
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 40),val,(0,0,0),font=f)
    charimg = np.array(img)
    charimg=255-charimg
    charimg=cv2.cvtColor(charimg,cv2.COLOR_BGR2GRAY)
    myimg,contours,h=cv2.findContours(charimg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # print contours[1]
    allcont=[]
    for cont in contours:
        for point  in cont:
            allcont.append(point)
    bdrct=cv2.boundingRect(np.array(allcont))
    # print contours
    contimg=cv2.cvtColor(myimg.copy(),cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contimg,contours,-1,(0,0,255))
    cv2.rectangle(contimg,(bdrct[0],bdrct[1]),(bdrct[0]+bdrct[2],bdrct[1]+bdrct[3]),(255,0,0))
    # print draw.textsize(val, font=f)

    if val in ['1']:
        centerx=bdrct[0]+0.5*bdrct[2]
        new_w=bdrct[2]*4
        leftx=int(centerx-0.5*new_w)
        rightx=int(centerx+0.5*new_w)
        boxchar = charimg[bdrct[1]:bdrct[1] + bdrct[3], leftx:rightx].copy()
        boxchar = cv2.resize(boxchar, (wid, hei), boxchar)
        return boxchar
    boxchar=charimg[bdrct[1]:bdrct[1]+bdrct[3],bdrct[0]:bdrct[0]+bdrct[2]].copy()
    boxchar=cv2.resize(boxchar,(wid,hei),boxchar)
    return boxchar
def gen_char_img(val,wid,hei,maskval):
    img=char_dict[val]
    img = cv2.resize(img, (wid, hei), img)
    cv2.threshold(img, 140, 255, 0, img)
    img=img/255*maskval
    return img

def gen_plate_char(tempdict,platestr):
    assert len(platestr)<=tempdict['numchar']
    # img = np.array(Image.new("RGB", (tempdict['width'], tempdict['height']),tempdict['bkcolor']))
    img=np.zeros(((tempdict['height'], tempdict['width'])))
    for i,val in enumerate(platestr):
        cur_rct=tempdict['char_rct'][i]
        hei=cur_rct[3]-cur_rct[1]
        wid=cur_rct[2]-cur_rct[0]
        # print hei,wid
        img[cur_rct[1]:cur_rct[3],cur_rct[0]:cur_rct[2]]=gen_char_img(val,wid,hei,i+1).copy()
    # dotxyr=tempdict['dotxyr']
    # cv2.circle(img, (dotxyr[0], dotxyr[1]), dotxyr[2], (255, 255, 255), thickness=-1)
    platestr_rct=tempdict['bdrct']
    return img,platestr_rct
def gen_plate_bk(tempdict):
    img = np.array(Image.new("RGB", (tempdict['width'], tempdict['height']), tempdict['bkcolor']))
    dotxyr=tempdict['dotxyr']
    cv2.circle(img, (dotxyr[0], dotxyr[1]), dotxyr[2],tempdict['chcolor'], thickness=-1)
    nailxyr=tempdict['nailxyr']
    for nail in  nailxyr:
        cv2.circle(img, (nail[0], nail[1]), nail[2], tempdict['chcolor'], thickness=-1)
    width=tempdict['width']
    height=tempdict['height']
    margin=4
    tlx=margin
    tly=margin
    brx=width-margin-1
    bry=height-margin-1
    chcolor=tempdict['chcolor']
    cv2.line(img,(tlx,tly),(brx,tly),chcolor,2)
    cv2.line(img, (brx, tly), (brx, bry), chcolor, 2)
    cv2.line(img, (brx, bry), (tlx, bry), chcolor, 2)
    cv2.line(img, (tlx, bry), (tlx,tly), chcolor, 2)
    return img

def get_stand_plate(plate_temp,platestr):
    charmask,platestr_rct=gen_plate_char(plate_temp, platestr)
    bkimg=gen_plate_bk(plate_temp)
    chcolor=np.array(plate_temp['chcolor'])
    for i in xrange(charmask.shape[0]):
        for j in xrange(charmask.shape[1]):
            if charmask[i,j]==0:
                continue
            bkimg[i,j,:]=chcolor
    for spcolor in plate_temp['sp_color']:
        for i in xrange(charmask.shape[0]):
            for j in xrange(charmask.shape[1]):
                if charmask[i,j]==spcolor[0]:
                    bkimg[i,j,:]=spcolor[1]
    return bkimg,platestr_rct

engnumtable='0123456789ABCDEFGHJKLMNPQRSTUVWXYZ-'
chi_table = u'京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼警港澳挂领使学字'

englist='ABCDEFGHJKLMNPQRSTUVWXYZ'
# randchrlist='012345678901234567890123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
# chilist = u'京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
fontC =  ImageFont.truetype("./font/platech.ttf",200,0);
fontE =  ImageFont.truetype('./font/platechar.ttf',200,0);
char_dict=gen_char_dict(fontC,fontE)

