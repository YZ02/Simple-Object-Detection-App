import cv2

#img = cv2.imread('lena.PNG')
cap = cv2.VideoCapture(0)    #create object for camera
#cap.set(3,640)               #set size of image/camera capturing
#cap.set(4,480)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)  #cv2 has pre-made model, we use DetectionModel (object detection)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean([127.5,127.5,127.5])
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.5) #if it's more than 50% then it is an object if not ignore

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0,255,0), thickness=2)  #make the bounding box (bbox) green
            cv2.putText(img, classNames[classId-1].upper(), (box[0]+10,box[1]+30),  #put text label, change its x,y position
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) #font type,scale,colour,thickness


    cv2.imshow("Output",img)
    cv2.waitKey(30)

#+-