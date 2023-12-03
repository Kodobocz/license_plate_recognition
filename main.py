import cv2
import pytesseract
import imutils


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Shows the image, and waits for a key to be pressed
def show_img(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)


while True:
    # Reading and resizing image
    image_number = input("Enter the number of the image: ")
    if not image_number.isdigit():
        break
    elif int(image_number) < 1 or int(image_number) > 24:
        break

    file_path = f"./data/image{image_number}.jpg"
    originalimage = cv2.imread(file_path)
    originalimage = imutils.resize(originalimage, width=500)
    show_img("Original Image", originalimage)


    # GrayScaling the image
    gray_image = cv2.cvtColor(originalimage, cv2.COLOR_BGR2GRAY) #->Extracting the luminance of the colors, so we only have the intensity of light in black and white
    show_img("Gray Scale Image", gray_image)

    # Smoothing the image to "ignore" unimportant contours and getting rid of noise
    gray_image_smooth = cv2.bilateralFilter(gray_image, 11, 17, 17)
    show_img("Smoother Image", gray_image)


    # Getting the edges
    edged_image = cv2.Canny(gray_image_smooth, 100, 150) #under 100-> edge if connected to two strong edges, over 150->strong edges
    show_img("Canny edge", edged_image)

    # Finding the contours
    cnts, new = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #CHAIN.. ->conserves only the end points of a straight edge

    # Drawing the contours on the main image
    contoured_image = originalimage.copy()
    cv2.drawContours(contoured_image, cnts, -1, (0, 255, 0), 3)
    show_img("Canny After Contouring", contoured_image)

    # Sorting the top 30 contourareas
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCount = None

    # Drawing these on the main image
    sorted_contoured_image = originalimage.copy()
    cv2.drawContours(sorted_contoured_image, cnts, -1, (0, 255, 0), 3)
    show_img("Top 30 Contours", sorted_contoured_image)

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
    show_img("Final Image", originalimage)

    # Cutting out the number plate
    crop_img_loc = f"./result/result_{image_number}.png"
    show_img("Cropped Image", cv2.imread(crop_img_loc))

    license_plate = cv2.imread(crop_img_loc)

    # Grayscaling the license plate
    grayscaled_license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    show_img("Grayscaled plate", grayscaled_license_plate)

    # Reading the text from the preprocessed license plate. specifying the language and characters the plate migth contains
    plate_text = pytesseract.image_to_string(grayscaled_license_plate ,config='--oem 3 -l eng --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    if len(plate_text.strip()) == 8:
        plate_text = plate_text[1:]

    # Check if the result's length is valid
    if len(plate_text.strip()) >= 6:
        print("Number Plate Text:", plate_text)
    else:
        print("Did not recognize any text on the license plate.")


    cv2.destroyAllWindows()


