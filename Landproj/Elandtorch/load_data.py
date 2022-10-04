
import data.inpainting_dataset as datasets
import cv2
import torchvision.transforms as transforms
import numpy as np
to_tensor = transforms.ToTensor()
import data.eland_data as datasets

def to_np_image(input):
    return np.array(transforms.ToPILImage()(input))

if __name__ == '__main__':
    faceattdata=datasets.ElandDataset('/disks/disk1/Dataset/Project/SuperResolution/taobao_stand_face')

    for i in range(0,100):
        data_dict=faceattdata[i]
        img=data_dict['input']


        img=to_np_image(img)

        cv2.imshow('img',img)
        key=cv2.waitKey(0)
        if key==27:
            exit(0)