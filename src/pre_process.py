import cv2 as cv
import os

img_path = "/home/zhao/2021/hat-detection/hat-detect/data/hat_label/test/0"

def pre_process(file,num):
    img = cv.imread(file)
    img_high = img.shape[0]
    img_width = img.shape[1]
    print('high:' + str(img.shape[0]))
    print('width:' + str(img.shape[1]))
    cut_img = img[0:int(img_high / 3), 0:int(img_width)]
    filename = '/home/zhao/2021/hat-detection/hat-detect/data/hat_label/test_half/0/out_img'+str(num)+'.jpg'
    cv.imwrite(filename,cut_img)

    '''
    cv.imshow('cut_img', cut_img)
    cv.waitKey(0)
    '''
def handle_path():
    list = os.listdir(img_path)
    for i in range(0,len(list)):
        print(list[i])
        file_path = os.path.join(img_path,list[i])
        pre_process(file_path,i)

handle_path()
