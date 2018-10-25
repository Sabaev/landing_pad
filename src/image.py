#! image processing
import cv2
import numpy as np
import src.image_loader as img_loader
import src.constants as const


def resize_img(img, width):
    ratio = float(width) / img.shape[1]
    return cv2.resize(img, (width, int(img.shape[0] * ratio)), interpolation=cv2.INTER_AREA)


def led_thresh(img, adapt_mask=False):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # warning adapt_mask is very slow
    if adapt_mask:
        GAUSSIAN_AREA = (img.shape[0] + img.shape[1]) // 4 + 1
        mask = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, GAUSSIAN_AREA, -70)
    else:
        ret, mask = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


# not effective
def get_point_by_detector(binary_img):
    par = cv2.SimpleBlobDetector_Params()
    par.blobColor = 255

    detector = cv2.SimpleBlobDetector_create(par)
    # Detect blobs.
    key_points = detector.detect(binary_img)

    return key_points


def find_contours(binary_image):
    # Находим контуры на изображении
    image, contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Если контуры не найдены, то ничего не возвращаем
    if not contours:
        return False, None
    return True, contours


def get_centers_from_contours(contours):

    centers = []

    for i in range(len(contours)):

        # Вычислеям моменты контура
        moments = cv2.moments(contours[i])

        # Вычисляем центр контура
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])

        centers.append([center_x, center_y])

    return centers


def get_points_by_moment(binary_img):
    res, contours = find_contours(binary_img)
    return get_centers_from_contours(contours) if res else None


def main():

    images = img_loader.load(const.path1)

    for i in images:

        i = resize_img(i, 1000)

        binary_img = led_thresh(i, adapt_mask=False)
        key_points = get_points_by_moment(binary_img)

        # draw circles
        for p in key_points:
            cv2.circle(i, (p[0], p[1]), 30, (0, 0, 255))
        # Вписываем центры в квадрат
        x, y, w, h = cv2.boundingRect(np.array(key_points))
        cv2.rectangle(i, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("image", i)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            cv2.destroyAllWindows()
            break
        if k == ord("n"):
            continue


if __name__ == '__main__':
    main()
