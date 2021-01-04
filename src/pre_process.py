import cv2 as cv

def pre_process(file):
    img = cv.imread(file)
    img_high = img.shape[0]
    img_width = img.shape[1]
    print('high:'+str(img.shape[0]))
    print('width:'+str(img.shape[1]))
    cut_img = img[0:int(img_high/3), 0:int(img_width)]
    cv.imshow('cut_img',cut_img)
    cv.waitKey(0)

pre_process('lena.jpg')