from imageai.Detection import ObjectDetection
import os


execution_path = os.getcwd()
image_src = "imagenew.jpg"
image_to_process = "image2.jpg"

detector = ObjectDetection()
# options to set detector to detect people only
people_only = detector.CustomObjects(person=True)
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(
    execution_path, "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(custom_objects=people_only, input_image=os.path.join(
    execution_path, image_to_process), output_image_path=os.path.join(execution_path, image_src))

for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"])
