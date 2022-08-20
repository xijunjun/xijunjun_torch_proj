
import data.inpainting_dataset as datasets
import cv2


if __name__ == '__main__':
    faceattdata=datasets.InpaintingDataset('/disks/disk1/Dataset/Project/SuperResolution/taobao_stand_face')

    for i in range(0,100):
        img=faceattdata[i]['input']
        print(img.shape)

        cv2.imshow('img',img)
        key=cv2.waitKey(0)
        if key==27:
            exit(0)