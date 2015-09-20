import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

PIT_DIST = 3
CFAT_RATIO = 2.54
WFAT_RATIO = 2.74

CM_TO_IN = 0.393701

def dist(p1, p2):
  return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))

def get_box(img):
  # b g r 
  # rgb(72, 141, 174) 174 141 72
  # rgb(40, 99, 131) 131 99 40
  # rgb(81, 187, 229) 229 187 81
  mask = cv2.inRange(img, np.array([120,80,30], np.uint8), np.array([240,190,90], np.uint8))

  cv2.imshow('blue2', mask)

  max_area = 0
  m_cnt = None
  contours, heirachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  for h, cnt in enumerate(contours):
    c_area = cv2.contourArea(cnt)

    if c_area > max_area:
      max_area = c_area
      m_cnt = cnt

  cv2.drawContours(img,[m_cnt],0,(0,0,255),3)

  rect = cv2.minAreaRect(m_cnt)
  return rect


src = cv2.imread('test8.jpg')
face_src = src.copy()

img = src.copy()

out = src.copy()

cv2.imshow('src', src)

rect = get_box(src)
box = cv2.cv.BoxPoints(rect)
box = np.int0(box)
cv2.drawContours(out,[box],0,(0,0,255),3)


box_diag = dist(box[0], box[2])

real_diag = dist((0,0), (10,8.3))

ratio = box_diag / real_diag


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.GaussianBlur(img, (3,3), 0)

img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 511 , 10)

cv2.imshow('blur', img)

# img = cv2.Canny(img, 5, 10)

# cv2.imshow('canny', img)

lwaist = 0

rows = img.shape[0]
cols = img.shape[1]

# Get WAIST
lwaist = 0
while img[rows*7/10][lwaist] == 255:
  lwaist = lwaist + 1

rwaist = cols-1
while img[rows*7/10][rwaist] == 255:
  rwaist = rwaist - 1

cv2.line(out, (lwaist, rows*7/10), (rwaist, rows*7/10), (255,0,0), 3)

# Get ARMPIT
lpit_x = lwaist
lpit_y = rows*7/10

while True:
  lpit_y -= 1
  count = 0
  while img[lpit_y][lpit_x] == 0 and count < PIT_DIST:
    count += 1
    lpit_x -= 1
  if count >= PIT_DIST:
    break

rpit_x = rwaist
rpit_y = rows*7/10

while True:
  rpit_y -= 1
  count = 0
  while img[rpit_y][rpit_x] == 0 and count < PIT_DIST:
    count += 1
    rpit_x += 1
  if count >= PIT_DIST:
    break

cv2.line(out, (lpit_x, lpit_y), (rpit_x, rpit_y), (255,0,0), 3)

waist_p = rwaist - lwaist
waist_cm = waist_p / ratio * WFAT_RATIO

print "Waist: %f" % waist_cm

cv2.putText(out, "Waist: %.2f\"" % (waist_cm * CM_TO_IN), (lwaist-150, rows*7/10+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

chest_p = dist((lpit_x, lpit_y), (rpit_x, rpit_y))
chest_cm = chest_p / ratio * CFAT_RATIO

cv2.putText(out, "Chest: %.2f\"" % (chest_cm * CM_TO_IN), (lwaist-150,rows*7/10-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

print "Chest: %f" % chest_cm

# Get FACE
faceCascade = cv2.CascadeClassifier('haar.xml')

faces = faceCascade.detectMultiScale(
  face_src,1.3,5
)

#print len(faces)
face = faces[0]

face_width_p = face[2]
face_height_p = face[3]

face_width_cm = face[2] / ratio
face_height_cm = face[3] / ratio

print "Face width: %.2fcm" % face_width_cm
print "Face height: %.2fcm" % face_height_cm

cv2.putText(out, "Face Width: %.2f\"" % (face_width_cm * CM_TO_IN), (face[0]-200, face[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
cv2.putText(out, "Face Height: %.2f\"" % (face_height_cm * CM_TO_IN), (face[0]-200, face[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

cv2.rectangle(out, (face[0], face[1]), (face[0]+face[2], face[1]+face[3]), (255, 0, 0), 3)

lneck = 0
neck_y = face[1]+face[3]
while img[neck_y][lneck] == 255:
  lneck = lneck + 1

rneck = cols-1
while img[neck_y][rneck] == 255:
  rneck = rneck - 1

cv2.line(out, (lneck, neck_y), (rneck, neck_y), (0,255,0), 3)

neck_p = rneck - lneck
neck_cm = neck_p / ratio * 2

print "Neck: %f" % neck_cm

cv2.putText(out, "Neck Height: %.2f\"" % (neck_cm * CM_TO_IN), (face[0]-200, face[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

cv2.imwrite('bryan.jpg', out)

cv2.waitKey(0)



# Kelvin
# 35.42 - 95.25
# 29.33 - 82.82

# Bryan
# 34.64 - 71.12
# 31.04 - 81.28

# Simon
# 33.07 - 97.79
# 28.3 - 78.74


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
