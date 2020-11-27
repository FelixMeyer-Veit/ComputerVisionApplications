################################################ IMPORT ################################################################
import cv2
import time
import numpy as np

# Import keras2onnx and onnx
#import onnx
#import keras2onnx
import matplotlib.pyplot as plt



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
    cv2.putText(frame, str(average), (1, 479), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

    return prev_frame_time

def init_classify(model='Model/flowers_model.onnx'):
    # Set global variables
    global net, classes

    # Define Class names
    classes = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
    classes.sort()

    # Read the model
    net = cv2.dnn.readNetFromONNX(model)


def classify_flower(image, size=1):
    # Pre-process the image
    img = cv2.resize(image, (200, 200))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array([img]).astype('float64') / 255.0

    net.setInput(img)
    Output = net.forward()  # only 1 output leads to better FPS

    # Post-process the results
    index = np.argmax(Output[0])
    prob = np.max(Output[0])

    label = classes[index]
    text = "{} {:.2f}%".format(label, prob * 100)

    cv2.putText(image, text, (5, size * 26), cv2.FONT_HERSHEY_COMPLEX, size, (100, 20, 255), 3)

    return image, (label, prob)

################################################ DECLARATIONS ##########################################################
# define a video capture object (webcam)
vid = cv2.VideoCapture(0)
vid.open("http://192.168.10.106:8080/video") # use android phone with app IP webcam

### for FPS
# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0
# create array for getting average
lst = []

###
# Load the keras model
#model = load_model('Model/flowers.h5')
# Convert it into onnx
#onnx_model = keras2onnx.convert_keras(model, model.name)
# Save the model as flower.onnx
#onnx.save_model(onnx_model, 'flowers_model.onnx')
################################################ MAIN ##################################################################
# Initialize the Network
init_classify()

while True:
    # Input image
    ret, frame = vid.read()
    result, tuple = classify_flower(frame, size=2)

    result = cv2.resize(result, (640, 480))

    # get fps and store a new prev_frame_time
    prev_frame_time = get_fps(result, prev_frame_time)

    # press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # Show the image with a rectangle surrounding the detected objects
    cv2.imshow('Image', result)

# Driver Code
average = Average(lst)

# Printing average of the list
print("Average processing time =", round(average, 2))

cv2.waitKey()
cv2.destroyAllWindows()
