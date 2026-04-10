import cv2
from ultralytics import YOLO

r = YOLO('best.pt')

re = r('img.png')

# print(cv2.imshow('img',re[0].plot()))
for i in re[0].boxes:
    print(f'class: {i.cls[0]}, conf: {i.conf}')
cv2.waitKey(0)