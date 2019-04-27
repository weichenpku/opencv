import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt


'''
Load and save images
'''
# load images
img = cv2.imread("images/Lenna.png", cv2.IMREAD_COLOR)
# img = cv2.imread("Lenna.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("Lenna.png", cv2.IMREAD_UNCHANGED)

# conversion between color and grayscale images
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# type and shape of images
print(type(img))
print(img.shape)

# show images
cv2.imshow("Lenna", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save images
cv2.imwrite("test.png", img)


'''
Draw and write on image
'''
# draw a line
cv2.line(img, (0, 0), (64, 64), (255, 0, 0), 10)

# multiple line
points = np.array([[350, 380], [380, 390], [390, 350], [350, 350]], dtype=np.int32)
cv2.polylines(img, [points], True, (255, 255, 0), 10)

# draw a rectangle
cv2.rectangle(img, (128, 128), (192, 192), (0, 255, 0), 10)

# draw a circle
cv2.circle(img, (256, 256), 64, (0, 0, 255), 10)

# mask images
img[400:450, 400:450] = 0
img[450:500, 450:500] = 255


'''
Chaning color space
'''
frame = cv2.imread("images/blue.jpg", cv2.IMREAD_COLOR)

# Convert BGR to HSV(Hue, Saturation, Value)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame, frame, mask=mask)

cv2.imshow('frame',frame)
cv2.imshow('mask',mask)
cv2.imshow('res',res)

'''
Image Operation
'''
# crop image
img_crop1 = img[200:300, 250:350]
img_crop2 = img[100:200, 350:450]
cv2.imshow("crop1", img_crop1)
cv2.imshow("crop2", img_crop2)

# arithmetics operation on images
diff = img_crop1 - img_crop2
cv2.imshow("diff", diff)

# logic operation
mask = (diff > 10).astype(np.uint8)
cv2.imshow("mask", mask * 255)

# assign pixel
img[100:200, 350:450] = diff
cv2.imshow("Lenna", img)

'''
Threshold
'''
# load images
img = cv2.imread("thrs.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("img", img)

thresh = 127
maxValue = 255
# global threshold
_, img_thrs_bin = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY);
cv2.imshow("img_thrs_bin", img_thrs_bin)

# adaptive threshold
img_thrs_mean = cv2.adaptiveThreshold(img, maxValue, 
                        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
cv2.imshow("img_thrs_mean", img_thrs_mean)

img_thrs_gauss = cv2.adaptiveThreshold(img, maxValue, 
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
cv2.imshow("img_thrs_gauss", img_thrs_gauss)


'''
Geometric Transformations
'''
# Resize image
# specify resize ratio
res05 = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
# cv2.INTER_LINEAR cv2.NEAREST
cv2.imshow("0.5 x resize", res05)
# specify output size of image
height, width = img.shape[:2]
res025 = cv2.resize(img,(int(0.25*width), int(0.25*height)), interpolation = cv2.INTER_CUBIC)
cv2.imshow("0.25 x resize", res025)

# Translation
M = np.float32([[1, 0, 100],
                [0, 1, 50]])
img_trans = cv2.warpAffine(img, M, (height, width))
cv2.imshow("translated image", img_trans)

# Rotation
M = cv2.getRotationMatrix2D((height / 2, width / 2), 90, 1)
img_rot = cv2.warpAffine(img, M, (height, width))
cv2.imshow("rotated image", img_rot)

# Affine transformation
pts1 = np.float32([[50,50], [200,50], [50,200]])
pts2 = np.float32([[10,100], [200,50], [100,250]])
M = cv2.getAffineTransform(pts1, pts2)
img_affine = cv2.warpAffine(img, M, (height, width))
cv2.imshow("affine transformed image", img_affine)

# perspective transformation
pts1 = np.float32([[56,65], [368,52], [28,387], [389,390]])
pts2 = np.float32([[0,0], [300,0], [0,300], [300,300]])
M = cv2.getPerspectiveTransform(pts1, pts2)
img_pers = cv2.warpPerspective(img, M, (height, width))
cv2.imshow("perspective transformed image", img_pers)



'''
Filtering
'''
# filtering, i.e. 2D conmvolution
kernel = np.ones((5, 5), np.float32) / 25
img_filtered = cv2.filter2D(img, -1, kernel)
cv2.imshow("img filtered", img_filtered)

# mean blurring, in this case kernel = np.ones((5, 5), np.float32) / 25
img_blur = cv2.blur(img, (5,5))     # NOTE that img_blur == img_filted here
cv2.imshow("img_blur", img_blur)

# gaussian blurring, in this case kernel is defined by gaussian distribution
# only distance is considered
img_blur_gauss = cv2.GaussianBlur(img, (5,5), 0)
cv2.imshow("img_blur_gauss", img_blur_gauss)

# median filtering, the central pixel is replaced with median value of the window
img_blur_median = cv2.medianBlur(img, 5)    # 5 is window size
cv2.imshow("img median", img_blur_median)

# bilateral filtering, similar to gaussian filtering, 
# except that similarity of pixel is also considered
# 9 is diameter of each pixel neighborhood, 75 are filter sigma in the color/coordinate space
img_bilateral = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imshow("img bialateral", img_bilateral)


'''
Morphological Transformations
'''
# erosion and dilation is performed on binary image, so we need to binarize image
img = cv2.adaptiveThreshold(img, 255, 
                        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow("img threshold", img)

kernel = np.ones((5, 5), np.uint8)
# erosion
img_erode = cv2.erode(img, kernel, iterations=1)
cv2.imshow("img eroded", img_erode)

# dilation
img_dilate = cv2.dilate(img, kernel, iterations=1)
cv2.imshow("img dilated", img_dilate)

# opening, i.e., erosion followed by dilation
img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow("img opening", img_opening)

# closing, i.e., dilation followed by erosion
img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow("img closing", img_closing)


'''
Image Gradient
'''
gray = np.float32(img)

# 2: blockSize - It is the size of neighbourhood considered for corner detection
# 3: ksize - Aperture parameter of Sobel derivative used.
# 0.04: k - Harris detector free parameter in the equation.
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01*dst.max()] = [0, 0, 255]
cv2.imshow('dst', img)
'''
'''
# gradient of x axis
# kernel = [[-1, 0, 1],
#           [-1, 0, 1],
#           [-1, 0, 1]]
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
cv2.imshow("x grad", sobelx)

# gradient of y axis
# kernel = [[-1, -1, -1],
#           [0, 0, 0],
#           [1, 1, 1]]
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1 ,ksize=3)
cv2.imshow("y grad", sobely)

# laplacian derivatives
laplacian = cv2.Laplacian(img, cv2.CV_64F)
cv2.imshow("laplacian grad", laplacian)


'''
KNN
'''
# Feature set containing (x,y) values of 25 known/training data
trainData = np.random.randint(0,100,(25,2)).astype(np.float32)

# Labels each one either Red or Blue with numbers 0 and 1
responses = np.random.randint(0,2,(25,1)).astype(np.float32)

# Take Red families and plot them
red = trainData[responses.ravel()==0]
plt.scatter(red[:,0],red[:,1],80,'r','^')

# Take Blue families and plot them
blue = trainData[responses.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],80,'b','s')

# new sample
newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')

knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
ret, results, neighbours, dist = knn.findNearest(newcomer, k=3)

print("result: ", results,"\n")
print("neighbours: ", neighbours,"\n")
print("distance: ", dist)
plt.show()



'''
K-Means
'''
X = np.random.randint(25,50,(25,2))
Y = np.random.randint(60,85,(25,2))
Z = np.vstack((X,Y))

# convert to np.float32
Z = np.float32(Z)

# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now separate the data, Note the flatten()
A = Z[label.ravel()==0]
B = Z[label.ravel()==1]

# Plot the data
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()


'''
YOLO demo
'''
args = {
    "yolo": "/path/to/model",
    "image": "/path/to/image",      
    "confidence": 0.5,              # minimum bounding box confidence
    "threshold": 0.3,               # NMS threshold
}

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
 
# load YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]
 
# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()
 
# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

boxes, confidences, classIDs = [], [], []
# loop over each of the layer outputs
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence of the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
 
		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the size of the image, 
            # keeping in mind that YOLO actually returns the center (x, y)-coordinates 
            # of the bounding box followed by the boxes' width and height
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
 
			# use the center (x, y)-coordinates to derive the top and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
 
			# update our list of bounding box coordinates, confidences, and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
 
		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)
 
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
