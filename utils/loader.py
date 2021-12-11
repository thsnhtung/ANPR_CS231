import sys
sys.path.append(r'C:\Users\Asus\Desktop\DoAnCV\utils')
sys.path.append(r'C:\Users\Asus\Desktop\DoAnCV')
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams

from utils.general import (check_file, check_imshow, increment_path)



auto = False 
stride = 16
imgsz = 320

source = str(r'C:\Users\Asus\Desktop\DoAnCV\video\Video camera giaothong.mp4')


dataset = LoadImages(source, img_size= imgsz, stride= stride, auto = auto)
bs = 1  # batch_size

for path, img, img0, cap in dataset:
    boex

