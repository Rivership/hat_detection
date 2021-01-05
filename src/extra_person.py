from imageai.Detection import ObjectDetection
import os
import tensorflow as tf
import cv2 as cv
from time import gmtime,strftime

#限制GPU显存的使用
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

base_path =  "./"
execution_path = "/mnt/disk1/hat-detection/hat-detect/data"
output_path = "/mnt/disk1/hat-detection/hat-detect/output"
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(base_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
custom_objects = detector.CustomObjects(person=True)

num = 0
count = 0
list = os.listdir(execution_path) #列出文件夹下所有的目录与文件
for i in range(0, len(list)):
    path = os.path.join(execution_path, list[i])
    if os.path.isfile(path):
        capture = cv.VideoCapture(os.path.join(execution_path, list[i]))
        while(capture.isOpened()):
            ret,img = capture.read()
            if ret == False:
                break
            num = num + 1
            if (num % 30) == 0:
                out_img, output_objects_array = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=img, input_type="array", output_type="array")
                #截取图片
                for output_object in output_objects_array:
                    img_copy = img[output_object['box_points'][1]:output_object['box_points'][3],
                               output_object['box_points'][0]:output_object['box_points'][2]]
                    if img_copy.size == 0:
                        continue
                    count = count + 1
                    file_name =  './output/'+str(count)+'.jpg'
                    print(num)
                    print(output_object['box_points'])
                    cv.imwrite(file_name, img_copy)

