import cv2
import numpy as np
import math
import pickle
import tensorflow as tf
import os
import sqlite3, pyttsx3
from keras.models import load_model
from threading import Thread


x, y, w, h = 300, 100, 300, 300
is_voice_on = True


# Sound Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #TF_CPP_MIN_LOG_LEVEL:3 = INFO, WARNING, and ERROR messages are not printed
model = load_model('cnn_model_keras2.h5') #Loading Model


def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape


image_x, image_y = get_image_size()


def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))   #set_image_dim_ordering('tf') so batch first image dimension ordering
	#(batch,image_width,image_height,channel)
	return img


def keras_predict(model, image):

	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0] #Numpy array(s) of predictions. It will return list of numooy array as multiple inputs
	pred_class = list(pred_probab).index(max(pred_probab))  #Finding index i.e label of maximum probability in pred_probab
	return max(pred_probab), pred_class


def get_pred_text_from_db(pred_class):

	conn = sqlite3.connect("gesture_db.db")  #This API opens a connection to the SQLite database file.
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
	cursor = conn.execute(cmd)#This routine executes an SQL statement.
	#It creates an intermediate cursor object by calling the cursor method, then calls the cursor's execute method with the parameters given.
	#The database cursor characteristic of traversal makes cursors akin to the programming language concept of iterator.
	# We could not get string from cursor object implicitly, that's why we created an iterator row to chnage it
	for row in cursor:
		#print(len(row))
		return row[0]

def get_pred_from_contour(contour, thresh):
	height, width, channels = contour.shape

	x1, y1, w1, h1 = cv2.boundingRect(contour)  #Calculates the up-right bounding rectangle of a point set.
	#The function calculates and returns the minimal up-right bounding rectangle for the specified point set.

	save_img = thresh[y1:y1+h1, x1:x1+w1] #Cropping Image
	text = ""
	if w1 > h1:
		save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0)) #Forms a border around image
	elif h1 > w1:
		save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
		#Here (w1-h1)/2 is done to make image a square shape which matches gestures recorded
		#cv2.copyMakeBorder(src,top,bottom,left,right,borderType,value)

	pred_probab, pred_class = keras_predict(model, save_img)
	if pred_probab*100 > 65:                         #Prediction probability threshold
		text = get_pred_text_from_db(pred_class)
	return text



def get_img_contour_thresh(img):
	img = cv2.flip(img, 1)   #Flipping image

	cv2.rectangle(img, (300,0),(500,200),(255,0,0),0)
	crop_img = img[0:200,300:500]

    # convert to grayscale
	grey=cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)

    # applying gaussian blur
	value = (11, 11) #Value defines kernel shape
	blurred = cv2.GaussianBlur(grey,value,0) #Here sigma_X=sigma_Y=0(Gradients in x and y drection)
	 											#if they are zero, then they will be calculated from kernel)

	#blurred = cv2.medianBlur(blurred, 7)

    # thresholding: The function is typically used to get a bi-level (binary) image out of a grayscale image
	_, thresh1 = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	# cv2.THRESH_OTSU it automatically calculates a threshold value from image histogram for a bimodal image.
	# bimodal image is that image in which hsitogram has two peaks



	#cv2.imshow('Thresholded', thresh1)

	contours= cv2.findContours(thresh1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1]
	cnt = max(contours, key = lambda x: cv2.contourArea(x))

    # create bounding rectangle around the contour (can skip below two lines)
	x, y, w, h = cv2.boundingRect(cnt)
	cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # finding convex hull
	hull = cv2.convexHull(cnt)

    # drawing contours
	drawing = np.zeros(crop_img.shape,np.uint8)
	cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
	cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    # finding convex hull
	hull = cv2.convexHull(cnt, returnPoints=False)

    # finding convexity defects
	defects = cv2.convexityDefects(cnt, hull)
	count_defects = 0
	cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    # applying Cosine Rule to find angle for all defects (between fingers)
    # with angle > 90 degrees and ignore defects
	for i in range(defects.shape[0]):
		s,e,f,d = defects[i,0]

		start = tuple(cnt[s][0])
		end = tuple(cnt[e][0])
		far = tuple(cnt[f][0])

        # find length of all sides of triangle
		a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
		b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
		c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # apply cosine rule here
		angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # ignore angles > 90 and highlight rest with red dots
		if angle <= 90:
		    count_defects += 1
		    cv2.circle(crop_img, far, 1, [0,0,255], -1)
        #dist = cv2.pointPolygonTest(cnt,far,True)

        # draw a line from start to end i.e. the convex points (finger tips)
        # (can skip this part)
		cv2.line(crop_img,start, end, [0,255,0], 2)
		cv2.circle(crop_img,far,5,[0,0,255],-1)
	thresh1 = thresh1[y:y+h, x:x+w] #DOUBT STILL HERE NEED TO SEE

	return img, contours, thresh1

def say_text(text):
	if not is_voice_on:
		return
	while engine._inLoop:
		pass
	engine.say(text)
	engine.runAndWait()

def text_mode(cam):
	global is_voice_on
	text = ""
	word = ""
	count_same_frame = 0
	x_c=30
	y_c=240
	delete_frame=0
	while True:
		img = cam.read()[1]

		height, width, channels = img.shape

		#compare_text_1=""
		compare_text_1="clear"



		img, contours, thresh = get_img_contour_thresh(img)

		old_text = text
		if len(contours) > 0:
			contour = max(contours, key = lambda x: cv2.contourArea(x))
			if cv2.contourArea(contour) > 10000:
				text = get_pred_from_contour(contour, thresh)



				if (compare_text_1==text ) and len(word)>0:
					if delete_frame<15:
						delete_frame=delete_frame+1

					elif delete_frame>=15:
						word = word[:-1]
						word[:word.rfind(' ')]
						word.rsplit(' ', 1)[0]
						delete_frame=0



				if old_text == text  and (compare_text_1!=text ):
					count_same_frame += 1
				else:
					count_same_frame = 0


				if count_same_frame > 20 and (text!='' and (compare_text_1!=text)) :
					Thread(target=say_text, args=(text, )).start()
					if not (word.endswith(" ")):
						word=word+" "
					word = word + text + " "
					count_same_frame = 0

			elif cv2.contourArea(contour) < 2000:
				if word != '':

					Thread(target=say_text, args=(word, )).start()
					print("writing to file..........")
					print(word)
					with open('output.txt', 'a') as f:
						f.write(word+'\n')

				text = ""
				word = ""
		else:
			if word != '':

				Thread(target=say_text, args=(word, )).start()
			text = ""
			word = ""


		blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
		cv2.putText(blackboard, "Sign Language Gesture Recognition", (140, 50), cv2.FONT_HERSHEY_TRIPLEX, .75, (255, 0,255))
		cv2.putText(blackboard, "Predicted text- " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, .65, (255, 255, 0))


		cv2.putText(blackboard, word, (x_c, y_c), cv2.FONT_HERSHEY_TRIPLEX, .55, (250, 230, 230))
		if is_voice_on:
			cv2.putText(blackboard, "Voice on", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, .65, (255, 127, 0))
		else:
			cv2.putText(blackboard, "Voice off", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, .65, (255, 127, 0))
		res = np.hstack((img, blackboard))   #Concatenating images of same size
		cv2.imshow("Recognizing gesture", res)



		keypress = cv2.waitKey(1)



		if keypress == ord('q') or keypress == ord('c'):
			break



		if keypress == ord('v') and is_voice_on:
			is_voice_on = False
		elif keypress == ord('v') and not is_voice_on:
			is_voice_on = True
		cv2.imshow("thresh", thresh)


	if keypress == ord('c'):
		return 2
	else:
		return 0

def recognize():
	cam = cv2.VideoCapture(1)
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	text = ""
	word = ""
	count_same_frame = 0
	keypress = 1
	while True:
		if keypress == 1:
			keypress = text_mode(cam)

		else:
			break

keras_predict(model, np.zeros((50, 50), dtype = np.uint8))
recognize()
