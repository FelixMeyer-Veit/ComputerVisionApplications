################################################ IMPORT ################################################################
# How to load a Tensorflow model using OpenCV
# Jean Vitor de Paulo Blog - https://jeanvitor.com/tensorflow-object-detecion-opencv/
import cv2
import time
import numpy as np

################################################ FUNCTIONS #############################################################
# get average of a list
def Average(lst):
    return sum(lst) / len(lst)

def get_fps(frame, prev_frame_time):
    # time when we finish processing for this frame
    new_frame_time = time.time()

    # Calculating the fps
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    # add current fps at the end of array
    lst.append(int(fps))

    # get the average of processing time
    average = int(Average(lst))

    # puting the FPS count on the frame
    cv2.putText(frame, str(average), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

    return prev_frame_time

def findObjects(outputs, frame):
    # Our operations on the frame come here
    rows, cols, channels = frame.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(detection[2]*cols), int(detection[3]*rows)
                x, y = int(detection[0]*cols - w/2), int(detection[1]*rows - h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    #print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nms_threshold=0.3)
    print("indices: " ,indices)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(frame, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

################################################ DECLARATIONS ##########################################################
# define a video capture object (webcam)
vid = cv2.VideoCapture(0)
vid.open("http://192.168.10.106:8080/video") # use android phone with app IP webcam

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

# create array for getting average
lst = []

img_counter = 0

classesFile = 'coco.names'
classNames = []

confThreshold = 0.5
whT = 320 # width-height-target
################################################ MAIN ##################################################################
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
print(len(classNames))

# Load a model imported from Tensorflow
#tensorflowNet = cv2.dnn.readNetFromTensorflow(('frozen_inference_graph.pb', 'graph.pbtxt'))

modelConfiguration = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'

# Results for 320:
# 0.98 fps
# knife: 96%, fork: 99% - spoon sometimes 55-60%: fork
# cup: 98%
# laptop as TVmonitor: 99%
# mouse as laptop: 99%
# bottle after taking off its bag: 50-80%, often times not detected
# person often times with 70-90% detected
# precision = 7/8
# recall = 7/10
# precision is a bit worse compared to tiny however recall is way better thats way in total F1-score is definitely better


#Results for tiny:
# fps = 7.69
# spoon sometimes 55%
# knife not detected and fork only once with 50% detected
# mouse as laptop: 50-80%
# laptop as tvmonitor: 80-90%
# keyboard as remote sometimes: 60-70%
# no cups or bottle detected
# person sometimes with 70% detected
# (sometimes randomly fridge for paper, book and blocks stacked together)

# recall is not too good because of a lot of false negatives
# precision is however better, when something is called, then it is mostly it

# tiny useful for raspberry pi/nano, cpu BUT loss of accuracy
# 320 with gpu higher speed and higher accuracy since it is slowing down cpu/pi

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    # Input image
    ret, frame = vid.read()
    # Our operations on the frame come here
    rows, cols, channels = frame.shape

    # set Input to network
    blob = cv2.dnn.blobFromImage(frame, 1/255, (whT, whT), [0, 0, 0], 1, crop=False) # [0, 0, 0] for the mean
    net.setInput(blob)

    # get Names of output-layers
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames) # outputs is a list
    # and outputs[0] is a numpy-array with outputs[0].shape = (300,85)
    # 300 = number of bounding boxes
    # values needed for storing bounding box are x,y,widht,height (first for values)
    # fifth value = confidence that object is present within bounding box
    # other 80 values are probabilities/predictions of each of the classes
    #print(outputs[0][1])

    findObjects(outputs, frame)

    '''
    # Use the given image as input, which needs to be blob(s).
    tensorflowNet.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))

    # Runs a forward pass to compute the net output
    networkOutput = tensorflowNet.forward()

    # Loop on the outputs
    for detection in networkOutput[0, 0]:

        score = float(detection[2])

        if score > 0.2:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows

        # draw a red rectangle around detected objects
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
    '''
    # get fps and store a new prev_frame_time
    prev_frame_time = get_fps(frame, prev_frame_time)

    # press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # Show the image with a rectangle surrounding the detected objects
    cv2.imshow('Image', frame)

# Driver Code
average = Average(lst)

# Printing average of the list
print("Average processing time =", round(average, 2))


cv2.waitKey()
cv2.destroyAllWindows()