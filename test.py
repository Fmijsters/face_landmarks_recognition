import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math


cap = cv2.VideoCapture('TOP 10 GOALS - 2018 FIFA WORLD CUP RUSSIA (EXCLUSIVE).mp4')
lsd = cv2.createLineSegmentDetector(1)

while(cap.isOpened()):
                # Read the frame
	width = int(cap.get(3))  # float
	height = int(cap.get(4)) #
	# dlines = lsd.detect(image)

	ret, frame = cap.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	dlines = lsd.detect(gray)

	img = np.zeros((height,width,3), np.uint8)
	for dline in dlines[0]:
		x0 = int(round(dline[0][0]))
		y0 = int(round(dline[0][1]))
		x1 = int(round(dline[0][2]))
		y1 = int(round(dline[0][3]))

		dist = math.hypot(x1 - x0, y1 - y0)
		if dist > 0:
			cv2.line(img, (x0, y0), (x1,y1), (255, 255, 255), 3, cv2.LINE_AA)







	cv2.imshow("video",img)
	cv2.imshow("normal",frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
    

cap.release()
cv2.destroyAllWindows()



piliimage = Image.open('136.jpg')
image = cv2.imread("136.jpg",0)
image2 = cv2.imread("136.jpg")
# lowthresh = 50
# highthresh = 150
# edges = cv2.Canny(image,lowthresh,highthresh,apertureSize = 3)
# indices = np.where(edges != [0])
# coordinates = list(zip(indices[0], indices[1]))

width, height = piliimage.size

fig, ax = plt.subplots()
fig.set_size_inches(20, 15)

ax.xaxis.tick_top()

extent = (0, width, height, 0)


image3 = image2

dlines = lsd.detect(image)

img = np.zeros((height,width,3), np.uint8)
for dline in dlines[0]:
	x0 = int(round(dline[0][0]))
	y0 = int(round(dline[0][1]))
	x1 = int(round(dline[0][2]))
	y1 = int(round(dline[0][3]))

	dist = math.hypot(x1 - x0, y1 - y0)
	if dist > 15:
		cv2.line(img, (x0, y0), (x1,y1), (255, 255, 255), 3, cv2.LINE_AA)

# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray,20,29,0.04)
# dst = cv2.dilate(dst,None)
# ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
# dst = np.uint8(dst)

# img[dst>0.01*dst.max()]=[0,0,255]
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

(thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]

teller = 0
lines = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
for x in range(1, 200):
	x = x * 0.1 
	lines2 = cv2.HoughLinesP(
	    im_bw, 1, np.pi / x, threshold=400, minLineLength=500, maxLineGap=20)
	lines[teller] = lines2
	# if lines2 is not None:
	# 	print("Found one with ", x)
	teller = teller + 1

for x in range(len(lines)):
	if lines[x] is  None:
		continue
	for dline in lines[x]:
		x0 = int(round(dline[0][0]))
		y0 = int(round(dline[0][1]))
		x1 = int(round(dline[0][2]))
		y1 = int(round(dline[0][3]))
		dist = math.hypot(x1 - x0, y1 - y0)
		if dist > 15:
			cv2.line(img, (x0, y0), (x1,y1), (255, 0, 0), 3, cv2.LINE_AA)
# for x in range(0,height-1):
# 	for y in range(0,width-1):
# 		b,g,r = img[x,y][0],img[x,y][1],img[x,y][2]
# 		if b != 0 and b!=255 and g != 0 and g!=255 and r != 0 and r!=255:
# 			img[x,y] = [255,255,255]



im = ax.imshow(img, cmap=plt.cm.hot, origin='upper',  extent=extent)

ax.set_title('Edges')

plt.show()




















# minLineLength = 5000
# maxLineGap = 1
# edges = cv2.Canny(img,lowthresh,highthresh,apertureSize = 3)
# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
# for line in lines:
# 	for x1,y1,x2,y2 in line:
# 		cv2.line(image3,(x1,y1),(x2,y2),(0,255,0),2)




# gray_image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# drawn_img = lsd.drawSegments(image3,lsd.detect(gray_image)[0])
# edges = cv2.Canny(drawn_img,lowthresh,highthresh,apertureSize = 3)







# print(lines)


# lines = lsd(edges)
# for i in xrange(lines.shape[0]):
#     pt1 = (int(lines[i, 0]), int(lines[i, 1]))
#     pt2 = (int(lines[i, 2]), int(lines[i, 3]))
#     width = lines[i, 4]
#     cv2.line(image, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))









# print(coordinates)
# cv2.imshow("Hey",image)
# cv2.imshow("edges",edges)

# cv2.waitKey(0)
# plt.subplot(121),plt.imshow(image)
# plt.axis([0, 1, 1.1 * 2, 2 * 6])
# plt.xlabel('(X)')
# plt.ylabel('(Y)')
# x1 = np.linspace(0.0, 5.0)
# x2 = np.linspace(0.0, 2.0)
# y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
# y2 = np.cos(2 * np.pi * x2)
# plt.plot(x1, y1, 'ko-')
# plt.title('Edge detection')
# xmin, xmax, ymin, ymax = plt.axis()
# print(xmin)
# print(xmax)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.plot(120),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])