import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


piliimage = Image.open('136.jpg')
image = cv2.imread("136.jpg",0)
image2 = cv2.imread("136.jpg")
lowthresh = 50
highthresh = 150
edges = cv2.Canny(image,lowthresh,highthresh,apertureSize = 3)
indices = np.where(edges != [0])
coordinates = list(zip(indices[0], indices[1]))
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
# plt.show()
width, height = piliimage.size
# plt.figure(num=1, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')

fig, ax = plt.subplots()
fig.set_size_inches(20, 15)
# fig.savefig('test2png.png', dpi=100)

ax.xaxis.tick_top()

extent = (0, width, height, 0)

# minLineLength = 100
# maxLineGap = 10
# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
# for line in lines:
# 	for x1,y1,x2,y2 in line:
# 		cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

# print(lines)


# lines = lsd(edges)
# for i in xrange(lines.shape[0]):
#     pt1 = (int(lines[i, 0]), int(lines[i, 1]))
#     pt2 = (int(lines[i, 2]), int(lines[i, 3]))
#     width = lines[i, 4]
#     cv2.line(image, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))

lsd = cv2.createLineSegmentDetector(1)

#Detect lines in the image
lines = lsd.detect(image)[0] #Position 0 of the returned tuple are the detected lines
image3 = image2
dlines = lsd.detect(image)
for dline in dlines[0]:
	x0 = int(round(dline[0][0]))
	y0 = int(round(dline[0][1]))
	x1 = int(round(dline[0][2]))
	y1 = int(round(dline[0][3]))
	cv2.line(image2, (x0, y0), (x1,y1), (255, 255, 255), 5, cv2.LINE_AA)

gray_image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# print(lsd.detect(edges))

drawn_img = lsd.drawSegments(image3,lsd.detect(gray_image)[0])
edges = cv2.Canny(drawn_img,lowthresh,highthresh,apertureSize = 3)

im = ax.imshow(image2, cmap=plt.cm.hot, origin='upper',  extent=extent)
# ax.plot(x, y, 'o')

ax.set_title('Edges')

plt.show()