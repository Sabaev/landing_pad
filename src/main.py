#! testing image
import cv2
import numpy as np
import matplotlib.pyplot as plt


def findlamps(img):
    # GaussianThresh
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    GArea = (img.shape[0] + img.shape[1]) // 5 + 1
    # print(GArea, "gaussian area")
    adaptiveMask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, GArea, -111)
    bImg = cv2.bitwise_and(img, img, mask=adaptiveMask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mImg = cv2.morphologyEx(bImg, cv2.MORPH_OPEN, kernel, iterations=1)
    # mImg = cv2.erode(bImg, kernel, iterations=1)

    par = cv2.SimpleBlobDetector_Params()
    par.blobColor = 255

    detector = cv2.SimpleBlobDetector_create(par)
    # Detect blobs.
    keypoints = detector.detect(mImg)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(mImg, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for i in keypoints:
        cv2.circle(im_with_keypoints, (int(i.pt[0]), int(i.pt[1])), 4, (255, 0, 0))

    return im_with_keypoints


def main():
    # matplotlib doesn't work, there are no circles on the images
    img1 = cv2.imread("../resource/1.jpeg")
    img2 = cv2.imread("../resource/2.jpeg")
    img3 = cv2.imread("../resource/3.jpeg")
    img4 = cv2.imread("../resource/4.jpeg")
    img5 = cv2.imread("../resource/5.jpeg")

    imgs = (img1, img2, img3, img4, img5)

    cv2.imshow("res", findlamps(img3))

    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()