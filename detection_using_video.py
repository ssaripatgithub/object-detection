from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()
camera = cv2.VideoCapture(0)


def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME ", frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")


def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last second: ",
          average_output_count)
    print("------------END OF A SECOND --------------")


def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print("MINUTE : ", minute_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last minute: ",
          average_output_count)
    print("------------END OF A MINUTE --------------")


detector = VideoObjectDetection()
people_only = detector.CustomObjects(person=True)
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(
    execution_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()


video_path = detector.detectObjectsFromVideo(
    custom_objects=people_only,
    camera_input=camera,
    output_file_path=os.path.join(execution_path, "camera_detected_video"),
    frames_per_second=20, log_progress=True, minimum_percentage_probability=40)
