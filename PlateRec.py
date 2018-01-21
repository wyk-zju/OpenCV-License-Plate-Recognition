import cv2
import time
import argparse
import numpy as np

#---ArgParse initialization------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', required = True, help = "path to the video file")
args = vars(parser.parse_args())
#--------------------------------------------------------------END---

cap = cv2.VideoCapture(args["video"])

while (cap.isOpened()):
    ret, img = cap.read()

    #---Reading Image----------------------------------------------------
    #img = cv2.imread(args["video"])
    cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image", img)
    #--------------------------------------------------------------END---

    #---RGB to Gray scale conversion-------------------------------------
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    cv2.namedWindow("Gray Converted Image",cv2.WINDOW_NORMAL)
    cv2.imshow("Gray Converted Image",img_gray)
    #--------------------------------------------------------------END---

    #-Noise removal with iterative bilateral filter(removes noise while preserving edges)
    noise_removal = cv2.bilateralFilter(img_gray, 9, 75, 75)
    cv2.namedWindow("Noise Removed Image",cv2.WINDOW_NORMAL)
    cv2.imshow("Noise Removed Image",noise_removal)
    #-----------------------------------------------------------------------------END----

    #-Histogram equalisation for better results--------------------------
    #-Increase contrast of image
    equal_histogram = cv2.equalizeHist(noise_removal)
    cv2.namedWindow("After Histogram equalisation",cv2.WINDOW_NORMAL)
    cv2.imshow("After Histogram equalisation",equal_histogram)
    #--------------------------------------------------------------END---

    #-Morphological opening with a rectangular structure element
    #-Clear small white peaces, not number plate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    morph_image = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel, iterations=5)
    cv2.namedWindow("Morphological opening",cv2.WINDOW_NORMAL)
    cv2.imshow("Morphological opening",morph_image)
    #--------------------------------------------------------------END---

    # Image subtraction(Subtracting the Morphed image from the histogram equalised Image)
    # Differrence between equal_histogram and morph_image
    sub_morp_image = cv2.subtract(equal_histogram, morph_image)
    cv2.namedWindow("Subtraction image", cv2.WINDOW_NORMAL)
    cv2.imshow("Subtraction image", sub_morp_image)
    #------------------------------------------------------------------------------END---

    #-Thresholding the image---------------------------------------------
    ret,thresh_image = cv2.threshold(sub_morp_image, 0, 255, cv2.THRESH_OTSU)
    cv2.namedWindow("Image after Thresholding",cv2.WINDOW_NORMAL)
    cv2.imshow("Image after Thresholding",thresh_image)

    #--------------------------------------------------------------END---

    #-Applying Canny Edge detection--------------------------------------
    #-Detect edges on the thresh_image
    canny_image = cv2.Canny(thresh_image, 250, 255)
    cv2.namedWindow("Image after applying Canny",cv2.WINDOW_NORMAL)
    cv2.imshow("Image after applying Canny",canny_image)
    canny_image = cv2.convertScaleAbs(canny_image)
    #--------------------------------------------------------------END---

    #-dilation to strengthen the edges-----------------------------------
    #-make lines of edges more thicker - tolszcze
    kernel = np.ones((3,3), np.uint8)   # Creating the kernel for dilation, each num of arr = 1
    dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
    cv2.namedWindow("Dilation", cv2.WINDOW_NORMAL)
    cv2.imshow("Dilation", dilated_image)
    #--------------------------------------------------------------END---

    #-Finding Contours in the image based on edges-----------------------------------------------------
    #-cv2.CHAIN_APPROX_SIMPLE ==> Remove all unnecessary contours, like lines from cv2.RETR_TREE
    #-cv2.RETR_TREE ==> mode of the contour retrieval algorithm which scna dilated_image for contours
    new, contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    # Sort the contours based on area ,so that the number plate will be in top 10 contours
    screenCnt = None
    # loop over our contours
    buffcontor = np.array([[300, 100], [300, 150], [350, 150], [350, 100]])
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Approximating with 6% error
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:  # Select the contour with 4 corners
            screenCnt = approx
            break
        else:
            screenCnt = buffcontor
            break
        #NEED TRY TO CREATE LOOP FOR ALL AVAILIBLE CONTOURS WITH 4 CORNERS!!!
    #--------------------------------------------------------------END---------------------------END---

    #-Drawing the selected contour on the original image-----------------
    final = cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
    cv2.namedWindow("Image with Selected Contour",cv2.WINDOW_NORMAL)
    cv2.imshow("Image with Selected Contour",final)
    #--------------------------------------------------------------END---

    #-Masking the part other than the number plate-----------------------
    mask = np.zeros(img_gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)
    cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
    cv2.imshow("Final_image",new_image)
    #--------------------------------------------------------------END---



    # Histogram equal for enhancing the number plate for further processing
    y, cr, cb = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2YCrCb))
    # Converting the image to YCrCb model and splitting the 3 channels
    y = cv2.equalizeHist(y)
    # Applying histogram equalisation
    final_image = cv2.cvtColor(cv2.merge([y,cr,cb]),cv2.COLOR_YCrCb2RGB)
    # Merging the 3 channels
    cv2.namedWindow("Enhanced Number Plate",cv2.WINDOW_NORMAL)
    # Creating a Named window to display image
    cv2.imshow("Enhanced Number Plate",final_image)
    # Display image

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
