import cv2
import imutils


while True:
    #Reading and resizing image
    image_number = input("Enter the number of the image: ")
    if image_number.isdigit() == False:
        break
    elif int(image_number) < 1 or int(image_number) > 24:
        break

    file_path = f"./data/image{image_number}.jpg"
    originalimage = cv2.imread(file_path)
    originalimage = imutils.resize(originalimage, width=500)
    cv2.imshow("Original Image", originalimage)
    cv2.waitKey(0)

    # GrayScaling the image
    gray_image = cv2.cvtColor(originalimage, cv2.COLOR_BGR2GRAY) #->Extracting the luminance of the colors, so we only have the intensity of light in black and white
    cv2.imshow("Gray Scale Image", gray_image)
    cv2.waitKey(0)

    # Smoothing the image to "ignore" unimportant contours and getting rid of noise
    gray_image_smooth = cv2.bilateralFilter(gray_image, 11, 17, 17)
    cv2.imshow("Smoother Image", gray_image)
    cv2.waitKey(0)

    # Getting the edges
    edged_image = cv2.Canny(gray_image_smooth, 100, 150) #under 100-> edge if connected to two strong edges, over 150->strong edges
    cv2.imshow("Canny edge", edged_image)
    cv2.waitKey(0)

    # Finding the contours
    cnts, new = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #CHAIN.. ->conserves only the end points of a straight edge

    # Drawing the contours on the main image
    contoured_image = originalimage.copy()
    cv2.drawContours(contoured_image, cnts, -1, (0, 255, 0), 3)
    cv2.imshow("Canny After Contouring", contoured_image)
    cv2.waitKey(0)

    # Sorting the top 30 contourareas
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCount = None

    # Drawing these on the main image
    sorted_contoured_image = originalimage.copy()
    cv2.drawContours(sorted_contoured_image, cnts, -1, (0, 255, 0), 3)
    cv2.imshow("Top 30 Contours", sorted_contoured_image)
    cv2.waitKey(0)

    # Finding the number plate
    for cnt in cnts:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True) #approxpoly->finding shape
        if len(approx) == 4:
            NumberPlateCount = approx
            x, y, w, h = cv2.boundingRect(cnt)
            crp_img = originalimage[y-10 : y+h+5 , x-10 : x+w+5]
            cv2.imwrite('./result/result_'+str(image_number) + '.png', crp_img)
            break

    # Drawing the contour on the main image
    cv2.drawContours(originalimage, [NumberPlateCount], -1, (0, 255, 0), 3)
    cv2.imshow("Final Image", originalimage)
    cv2.waitKey(0)

    # Cutting out the number plate
    crop_img_loc = f"./result/result_{image_number}.png"
    cv2.imshow("Cropped Image", cv2.imread(crop_img_loc))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



