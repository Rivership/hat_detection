from imageai.Detection import ObjectDetection
import os
import tensorflow as tf

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

list = os.listdir(execution_path) #列出文件夹下所有的目录与文件
for i in range(0, len(list)):
       path = os.path.join(execution_path, list[i])
       if os.path.isfile(path):
           detections, objects_path = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path, list[i]), output_image_path=os.path.join(output_path, list[i]), extract_detected_objects=True)
           for eachObject, eachObjectPath in zip(detections, objects_path):
               print(eachObject["name"], " : ", eachObject["percentage_probability"])
               print("Object's image saved in " + eachObjectPath)
               print("--------------------------------")
