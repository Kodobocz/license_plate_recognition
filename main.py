import cv2
import imutils
import pytesseract

from functions import TestWaitKey

#Download and set route
#https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.3.0.20221214.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#Reading and resizing image
image = cv2.imread(".\data\image1.jpg")
image = imutils.resize(image, width=500)
cv2.imshow("Original Image", image)
TestWaitKey()

# GrayScaling the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Scale Image", gray)
TestWaitKey()

# Smoothing the image to ignore unimportant contours
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("Smoother Image", gray)
TestWaitKey()

# Make edges more defined  -->szakadozottsághoz állít, alacsony treshold alatti off, 2treshold között akkor, ha összeköt 2 erős kontúrt, semmi közepén eldob
edged = cv2.Canny(gray, 100, 150)
cv2.imshow("Canny edge", edged)
TestWaitKey()

# Finding the contours
cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Drawing the contours on the main image
image1 = image.copy()
cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Canny After Contouring", image1)
TestWaitKey()

# Sorting the top 30 most "contoured" places
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
NumberPlateCount = None

# Drawing these on the main image
image2 = image.copy()
cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Top 30 Contours", image2)
TestWaitKey()

count = 0
name = 1

# Finding the number plate
for cnt in cnts:
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True) #approxpoly->finding shape
    if len(approx) == 4:
        NumberPlateCount = approx
        x, y, w, h = cv2.boundingRect(cnt)
        crp_img = image[y-10 : y + h+5, x-10 : x + w+5]
        cv2.imwrite(str(name) + '.png', crp_img)
        name += 1

        break


# Drawing the contour on the main image
cv2.drawContours(image, [NumberPlateCount], -1, (0, 255, 0), 3)
cv2.imshow("Final Image", image)
TestWaitKey()

# Cutting out the number plate
crop_img_loc = '1.png'
cv2.imshow("Cropped Image", cv2.imread(crop_img_loc))
TestWaitKey()