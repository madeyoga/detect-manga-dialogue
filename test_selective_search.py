import cv2
from selectivesearch import selective_search

img = cv2.imread('samples/sample1.png')

cv2.imshow('og', img)

img_lbl, regions = selective_search(img, scale=500, sigma=0.9, min_size=10)

len(regions)
for r in regions:
    (x,y,w,h) = r['rect']
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 1)

cv2.imshow('res', img)
