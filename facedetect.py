import cv2
import sys

#We will first tell the python script (code) the images and cascade we are gonna use. 
#we will pass the as command-line arguments. 
imagePath = sys.argv[1]
cascPath = sys.argv[2]

#after creating the cascade now we will initialize it with a variable faceCascade.
#basically we are loading the faceCascade into the memory so that we can use the XML file.
faceCascade = cv2.CascadeClassifier(cascPath)

#read the image and convert it to gray scale. 
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print "Found {0} faces!".format(len(faces))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
           
cv2.imshow("result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
