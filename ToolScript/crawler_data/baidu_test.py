from BaiduImagesDownload import Crawler

# original为True代表优先下载原图
net, num, urls = Crawler.get_images_url('二次元', 20, original=True)

Crawler.download_images(urls,path='/home/tao/disk1/Dataset/Project/FaceEdit/half_head_hair/originimage/baidu_crawler')

