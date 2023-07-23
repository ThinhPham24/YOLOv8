# Custom deep learning developmentation 
import os
import numpy as np
import glob
from tqdm import tqdm
import cv2
import time
# Input folder
folder = '/Shuffle_Images_v1'
current_dir = os.getcwd()
path = ''.join([current_dir, folder])
print("path = ", path)
folder_out = '/OUTPUT'
path_out = ''.join([current_dir, folder_out])
if not os.path.isdir(os.path.abspath(path_out)):
        os.mkdir(path_out)

if __name__ == '__main__':
        model = cv2.dnn.readNet(model=f'{current_dir}\\best_yolov8.onnx', config=f'{current_dir}\\orchid_config.yaml', framework='onnx')
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        path_img = glob.glob(path + '/' + '*.jpg')
        
        for i, image in enumerate(path_img):
                t1 = time.time()
                print("image:", image)
                img = cv2.imread(image)
                blob = cv2.dnn.blobFromImage(img, 1, (736,736))
                # set the input blob for the neural network
                model.setInput(blob)
                # forward pass image blog through the model
                outputs = model.forward()
                # outputs = outputs.transpose((0, 2, 1))
                print("result:", outputs[0])
                cv2.imshow("image", img)
                t2 = time.time()
                print("All time", (t2-t1))
                k = cv2.waitKey(0)
                if k == ord("q"):
                        break


