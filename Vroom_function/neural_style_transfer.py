import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import time
import os
import cv2
import numpy as np

global net

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')


print(os.getcwd())

def load_model(img_path):
        # Yolo 로드
    
    
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))


    # 이미지 가져오기
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape


    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
        # 정보를 화면에 표시
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x - 70, y + 40), font, 3, color, 3, cv2.LINE_AA)
    
    cv2.imwrite(img_path,img)        
    
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    test = None
    for i in class_ids:
        test = i
    
    return test



def main(target_img_path):
    # image 파일명
    target_img_path = target_img_path.split('/')[-1]
    # image 경로
    target_image_path = './static/images/'+ target_img_path 			# 타깃 이미지 
    
    test = load_model(target_image_path)

    # fname = '../flask_deep/static/images/nst_get/' + 'new.jpg'

    # save_img(fname, img)

    return test

if __name__ == "__main__":
	main()