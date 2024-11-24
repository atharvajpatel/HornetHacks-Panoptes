import cv2
import numpy as np

# Load image
img = cv2.imread("celestial_navigation/example_input.jpg")

# Convert to grayscale image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Set a threshold for brightness (here, set it to 150). originally 
ret, new = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

# The image is made up of white dots on a black background, so the outlines of the white dots are detected.
#detected_image, 
contours, hierarchy = cv2.findContours(new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# For each contour, find the center of gravity and put it into an array
stars = []
for cnt in contours:
    M = cv2.moments(cnt)
    if M['m00'] != 0: # if area is non-zero:
        cx = int(M['m10'] / M['m00']) # m10 is the sum of x pixels in the contour. normalize by area
        cy = int(M['m01'] / M['m00'])
        # centroid coordinates
        stars.append(np.array([cx, cy], dtype='int32'))
    else:
        # If the moment calculation fails, take the first point of the contour
        stars.append(np.array(cnt[0][0], dtype='int32'))

# Return or print the array of star positions
#print(stars)

img_with_contours = img.copy()
cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)  # Green contours with thickness 2

cv2.imshow("Image with Contours", img_with_contours)
cv2.waitKey(0) # press 0 to exit
cv2.destroyAllWindows()