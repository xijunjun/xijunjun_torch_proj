#python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \
#    --network=./checkpoints/metfaces.pkl \
#    --outdir=./out

#python style_mixing.py --outdir=out --rows=85,100,75,458,1500 --cols=55,821,1789,293 \
#    --network=./checkpoints/metfaces.pkl  \
#    --outdir=./out

#python test_net.py --outdir=out --trunc=1 --seeds=85,265,297,849 \
#    --network=./checkpoints/ffhq.pkl \
#    --outdir=./out

#python style_mixing.py --outdir=out --rows=85,100,75,458,1500 --cols=55,821,1789,293 \
#    --network=./checkpoints/ffhq.pkl

python style_mixing.py --outdir=myout --rows=85,100,75,458,1500 --cols=55,821,1789,293 \
    --network=/disks/disk1/Workspace/Project/Pytorch/CheckPoints/styleganada/00/00052-CelebA-HQ-img-256-matting-auto2-noaug-resumecustom/network-snapshot-000400.pkl