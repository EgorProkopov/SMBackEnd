import numpy as np
import cv2
import os

urlIn = r"C:\Users\DenisovDmitrii\Desktop\train\train"
urlOut = r"C:\Users\DenisovDmitrii\Desktop\train\trainTrue"
listFiles = os.listdir(urlIn)
for file in listFiles:
    image = cv2.imread(urlIn + "\\" + file, cv2.IMREAD_COLOR)
    bw_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rev, bw_img = cv2.threshold(bw_img, 1, 255, cv2.THRESH_BINARY)
    #bw_img = (bw_img == 255).astype(np.uint8)
    #print(bw_img.max())
    cv2.imwrite(urlOut + "\\" + file, bw_img)
    #input()

