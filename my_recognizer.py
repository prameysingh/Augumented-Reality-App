import os
import sys 
import numpy as np 
import cv2
from PIL import Image
import time

cascadePath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will use the the Local Binary Pattern Histogram Face Recognizer 
recognizer = cv2.createLBPHFaceRecognizer()



#print(moment)






def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad (the sad face) extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject",""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(frame,
    										scaleFactor=1.5,
    										minNeighbors=6,
    										minSize=(30, 30),
    										flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels
video_capture = cv2.VideoCapture(0)
for i in range(100):

	
	ret, frame = video_capture.read()
	ret = video_capture.set(3,500)
	ret = video_capture.set(4,500)
	      



	#moment=time.strftime("%Y%b%d%H%M%S",time.localtime())   
	minutes = int(time.strftime("%M",time.localtime()))
	seconds = int(time.strftime("%S",time.localtime()))	
	moments = minutes*60 + seconds
	moment = str(moments)

	path = './my_database'
	images, labels = get_images_and_labels(path)
	cv2.destroyAllWindows()

	labels_list = list(labels)
	length = len(labels_list)
	
	#for i in range(length-1):
	#	print(labels[i])



	recognizer.train(images, np.array(labels))

	video_image = np.array(frame, 'uint8')
	im = Image.fromarray(video_image)
	im.save("/home/pranav_sankhe/Documents/my_projects/Augumented-Reality-App-master/my_database/"+"subject"+moment+"."+"jpg")
	image_paths = ["/home/pranav_sankhe/Documents/my_projects/Augumented-Reality-App-master/my_database/"+"subject"+moment+"."+"jpg"]
	for image_path in image_paths:
    		predict_image_pil = Image.open(image_path).convert('L')
    		predict_image = np.array(predict_image_pil, 'uint8')
    		faces = faceCascade.detectMultiScale(frame,
    									scaleFactor=1.5,
    									minNeighbors=6,
    									minSize=(30, 30),
    									flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    	for (x, y, w, h) in faces:
        	nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
        	nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        	print("number :  ")
        	print(nbr_actual)
        	if nbr_actual == nbr_predicted:
        		print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
        	else:
        		print "{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted)
        	cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
        	


