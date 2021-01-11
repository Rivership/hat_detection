import cv2 as cv
from time import gmtime,strftime
import os

#从视频中读取文件，转换成image再转换成array
data_read_path = "/mnt/disk1/hat-detection/hat-detect"
def video_grab_frame(file):
    capture = cv.VideoCapture(file)
    while(capture.isOpened()):
        ret, frame = capture.read()
        if ret == False:
            print("finish")
            break
        f = strftime("%Y%m%d%H%M%S.jpg",gmtime())
        print(f)
        cv.imwrite('./data'+ f, frame)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

video_grab_frame(os.path.join(data_read_path,"001.MOV"))
