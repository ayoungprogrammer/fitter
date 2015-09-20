import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('test8.jpg')

cv2.imshow('blue', img)

# Threshold the HSV image to get only blue colors
# r g b
# b g r
mask = cv2.inRange(img, np.array([80,80,30], np.uint8), np.array([150,110,50], np.uint8))

cv2.imshow('blue2', mask)



max_area = 0
m_cnt = None
contours, heirachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for h, cnt in enumerate(contours):
  c_area = cv2.contourArea(cnt)

  if c_area > max_area:
    max_area = c_area
    m_cnt = cnt

#cv2.drawContours(img,[m_cnt],0,(0,0,255),3)

rect = cv2.minAreaRect(m_cnt)
box = cv2.cv.BoxPoints(rect)
box = np.int0(box)
cv2.drawContours(img,[box],0,(0,0,255),3)

cv2.imshow('blue3', img)

cv2.waitKey(0)


# img = cv2.GaussianBlur(img, (3,3), 0)

# #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 131 , 10)

# cv2.imshow('blur', img)

# img = cv2.Canny(img, 5, 10)

# cv2.imshow('canny', img)


# contours, heirachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

# maskc = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

# max_area = 0
# m_cnt = None

# for h, cnt in enumerate(contours):
#   c_area = cv2.contourArea(cnt)

#   if c_area > max_area:
#     max_area = c_area
#     m_cnt = cnt

# cv2.drawContours(maskc, [m_cnt], 0, (0, 255,0), 3)

# cv2.imshow('teststest', maskc)

# cv2.waitKey(0)

# plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.xticks([]), plt.yticks([])
# plt.show()
