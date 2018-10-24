#! testing image
import cv2
import numpy as np
import src.image_loader as img_loader


def resize_img(img, width):
    ratio = float(width) / img.shape[1]
    resized_img = cv2.resize(img, (width, int(img.shape[0] * ratio)), interpolation=cv2.INTER_LINEAR)
    return resized_img


def led_thresh(img):
    # GaussianThresh
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    G_AREA = (img.shape[0] + img.shape[1]) // 4 + 1

    # print(GArea, "gaussian area")
    gaussian_mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, G_AREA, -70)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    mask = cv2.morphologyEx(gaussian_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # res = cv2.bitwise_and(img, img,cv2.THRESH_BINARY, mask=mask)

    # mImg = cv2.erode(bImg, kernel, iterations=1)
    return mask


def led_detector(img):
    par = cv2.SimpleBlobDetector_Params()
    par.blobColor = 255

    detector = cv2.SimpleBlobDetector_create(par)
    # Detect blobs.
    key_points = detector.detect(img)

    return cv2.KeyPoint_convert(key_points)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # img_with_keypoints = cv2.drawKeypoints(mImg, key_points, np.array([]), (0, 0, 255),
    #                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # for i in keypoints:
    #     cv2.circle(img_with_keypoints, (int(i.pt[0]), int(i.pt[1])), 4, (255, 0, 0))

    # return img_with_keypoints


def main():
    path1 = "../resource/image/led/0"
    path2 = "../resource/image/led_with_noise"
    images = img_loader.load(path2, ".jpg")

    for i in images:

        i = resize_img(i, 1000)
        thresh_img = led_thresh(i)
        key_points = led_detector(thresh_img)
        cv2.imshow("morpho_mask", thresh_img)

        print(len(key_points))
        for p in key_points:
            cv2.circle(i, (p[0], p[1]), 30, (0, 0, 255))

        # Вписываем центры в квадрат
        x, y, w, h = cv2.boundingRect(key_points)
        cv2.rectangle(i, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("image", i)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break
        if k == ord("n"):
            continue


if __name__ == '__main__':
    main()
