#python train.py --outdir=/home/tao/disk1/Workspace/Project/Pytorch/CheckPoints/styleganada/00 --data=/home/tao/disk1/Dataset/CelebAMask/CelebAMask-HQ/CelebA-HQ-img-128 --gpus=2
#CUDA_VISIBLE_DEVICES=1 python train.py --outdir=/home/tao/disk1/Workspace/Project/Pytorch/CheckPoints/styleganada/00 --data=/home/tao/disk1/Dataset/CelebAMask/CelebAMask-HQ/CelebA-HQ-img-256 \
#--resume=/disks/disk1/Workspace/Project/Pytorch/FaceEdit/stylegan2-ada-pytorch-main/checkpoints/ffhq-res256-mirror-paper256-noaug.pkl \
#--aug=noaug --gpus=1 --workers=1



#CUDA_VISIBLE_DEVICES=0,1 python train.py --outdir=/home/tao/disk1/Workspace/Project/Pytorch/CheckPoints/styleganada/00 --data=/home/tao/disk1/Dataset/CelebAMask/CelebAMask-HQ/CelebA-HQ-img-256-matting \
#--resume=/disks/disk1/Workspace/Project/Pytorch/FaceEdit/stylegan2-ada-pytorch-main/checkpoints/ffhq-res256-mirror-paper256-noaug.pkl \
#--aug=noaug --gpus=2 --workers=2


#CUDA_VISIBLE_DEVICES=1 python train.py --outdir=/home/tao/disk1/Dataset/CelebAMask/CelebAMask-HQ/CelebA-HQ-img-256-4c \
#--resume=/disks/disk1/Workspace/Project/Pytorch/FaceEdit/stylegan2-ada-pytorch-main/checkpoints/ffhq-res256-mirror-paper256-noaug.pkl \
#--aug=noaug --gpus=1 --workers=1


#CUDA_VISIBLE_DEVICES=1  python train.py --outdir=/home/tao/disk1/Workspace/Project/Pytorch/CheckPoints/styleganada/00 --data=/home/tao/disk1/Dataset/CelebAMask/CelebAMask-HQ/zip/CelebA-HQ-img-256.zip --gpus=1

#CUDA_VISIBLE_DEVICES=0,1 python train.py --outdir=/home/tao/disk1/Workspace/Project/Pytorch/CheckPoints/styleganada/00 --data=/home/tao/disk1/Dataset/CelebAMask/CelebAMask-HQ/CelebA-HQ-img-256-4c \
#--resume=/disks/disk1/Workspace/Project/Pytorch/FaceEdit/stylegan2-ada-pytorch-main/checkpoints/ffhq-res256-mirror-paper256-noaug.pkl \
#--aug=noaug --gpus=2 --workers=2


CUDA_VISIBLE_DEVICES=1 python train.py --outdir=/home/tao/disk1/Workspace/Project/Pytorch/CheckPoints/styleganada/00 --data=/home/tao/disk1/Dataset/CelebAMask/CelebAMask-HQ/CelebA-HQ-img-256-4c \
--resume=/disks/disk1/Workspace/Project/Pytorch/FaceEdit/stylegan2-ada-pytorch-main/checkpoints/ffhq-res256-mirror-paper256-noaug.pkl \
--aug=noaug --gpus=1 --workers=1