import numpy as np
import cv2 as cv

#打开摄像头获取图片
def video_demo():
    cap = cv.VideoCapture(0)
    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = int(cap.get(5))
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('./testwrite2.avi',fourcc,fps,(w,h),True)
    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret == True:
            cv.imshow('frame',frame)
            out.write(frame)

            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            print("read file fail.")
            break

video_demo()
cv.destroyAllWindows()