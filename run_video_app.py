# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from smbus2 import SMBus
from mlx90614 import MLX90614
import RPi.GPIO as GPIO
from time import sleep
import notify2
import subprocess
#import _thread
#import threading


#thread lock
#lock = threading.Lock()

#initialize temperature sensor bus and gpio
bus = SMBus(1)
sensor = MLX90614(bus, address=0x5a)

#LED setup
greenLed = 8
redLed = 7
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(greenLed, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(redLed, GPIO.OUT, initial=GPIO.LOW)

#Servo motor setup
servoPin = 15
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servoPin, GPIO.OUT)
pwm = GPIO.PWM(servoPin, 50)


#pwm.start(2.5)

#Buzzer setup
buzz = 11
GPIO.setup(buzz, GPIO.OUT)
GPIO.output(buzz, GPIO.LOW)

#IR sensor setup
ir = 10
GPIO.setup(ir, GPIO.IN)



def sendMessage(title, msg):
    subprocess.Popen(['notify-send', msg])
    return


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
		
		

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)


def openGate():
    pwm.ChangeDutyCycle(2.0)
    #pigpio.set_PWM_dutycycle(2.0)
    sleep(0.5)
    
    
def closeGate():
    pwm.ChangeDutyCycle(12.0)
    #pigpio.set_PWM_dutycycle(12.0)
    sleep(0.1)
    

#Apply Algorithm
def applyLogic(label):
    pwm.start(0)
    temp = getTempData()
    if temp >= 37:
        GPIO.output(buzz, GPIO.HIGH)
        GPIO.output(redLed, GPIO.HIGH)
        sleep(1)
    elif (label=="No Mask"):
        GPIO.output(redLed, GPIO.HIGH)
        GPIO.output(greenLed, GPIO.LOW)
        GPIO.output(buzz, GPIO.LOW)
        #gateClose = threading.Thread(target=closeGate)
        #gateClose.start()
        closeGate()
        sendMessage("No Mask","Please wear mask!")
    else:
        GPIO.output(buzz, GPIO.LOW)
        GPIO.output(greenLed, GPIO.HIGH)
        GPIO.output(redLed, GPIO.LOW)
        #gateOpen = threading.Thread(target=openGate)
        #gateOpen.start()
        openGate()
        

def getTempData():
    temp = sensor.get_object_1()
    return temp

def closeEverything():
    GPIO.output(redLed, GPIO.LOW)
    GPIO.output(greenLed, GPIO.LOW)
    GPIO.output(buzz, GPIO.LOW)
    closeGate()

def detect_mask(locs, preds, frame):
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
            
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        print(label)

        # include the probability in the label
        label_out = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        #temperature sensor data
        temp = getTempData()
        #temp = sensor.get_object_1()
        person_temp = "Temp: {:.1f}".format(temp)
        
        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label_out, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.putText(frame, person_temp, (endX-10, endY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        dist = GPIO.input(ir)

        if dist == 0:
            applyLogic(label)
        else:
            closeEverything()
        
        #_thread.start_new_thread(applyLogic, (label,))



# loop over the frames from the video stream
def run_video(detect_and_predict_mask):
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=1000)
        #cv2.normalize(frame, frame,0,255, cv2.NORM_MINMAX)
        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
            

        # loop over the detected face locations and their corresponding
        # locations
        detect_mask(locs, preds, frame)
        
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup

    cv2.destroyAllWindows()
    pwm.stop()
    GPIO.cleanup()
    vs.stop()
    


#main function
if __name__=="__main__":
    # load our serialized face detector model from disk
    prototxtPath = "/home/pi/Desktop/Project/face_detector/deploy.prototxt"
    weightsPath = "/home/pi/Desktop/Project/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    maskNet = load_model("mask_detector.model")
    
    #opening gate
    #gate = threading.Thread(target=openGate)

    # initialize the video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0, framerate=30).start()
    
    run_video(detect_and_predict_mask)

    
