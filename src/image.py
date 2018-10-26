#! image processing
import cv2
import numpy as np
import src.image_loader as img_loader
import src.constants as const
import math
from src.camera import Camera


class Image:

    def __init__(self, img, new_width=1000, resize=True):
        self.binary_img = None
        self.contours = None

        if resize:
            ratio = float(new_width) / img.shape[1]
            self.width = new_width
            self.hight = int(img.shape[0] * ratio)
            self.img = cv2.resize(img, (new_width, self.hight), interpolation=cv2.INTER_AREA)
        else:
            self.hight = img.shape[0]
            self.width = img.shape[1]
            self.img = img

    def get_points(self, moment_method=True):
        self.threshold()
        if moment_method:
            return self.get_points_by_moment()
        else:
            return self.get_point_by_detector()

    def threshold(self, adapt_mask=False):
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # warning adapt_mask is very slow
        if adapt_mask:
            GAUSSIAN_AREA = (gray_img.shape[0] + gray_img.shape[1]) // 4 + 1
            mask = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, GAUSSIAN_AREA, -100)
        else:
            ret, mask = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.binary_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    def get_points_by_moment(self):
        res, contours = self.get_contours()
        if res:
            return True, self.get_centers_from_contours()
        return False, None

    def get_contours(self):
        if self.binary_img is None:
            return False, None
        # Находим контуры на изображении
        image, self.contours, hierarchy = cv2.findContours(self.binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Если контуры не найдены, то ничего не возвращаем
        if not self.contours:
            return False, None
        return True, self.contours

    def get_centers_from_contours(self):
        if self.contours is None:
            return None

        centers = []

        for i in range(len(self.contours)):
            # Вычислеям моменты контура
            moments = cv2.moments(self.contours[i])

            # Вычисляем центр контура
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])

            centers.append([center_x, center_y])

        return centers

        # not effective

    def get_point_by_detector(self):
        if self.binary_img is None:
            return False, None
        par = cv2.SimpleBlobDetector_Params()
        par.blobColor = 255
        detector = cv2.SimpleBlobDetector_create(par)
        # Detect blobs.
        key_points = detector.detect(self.binary_img)
        return True, key_points


def main():
    images = img_loader.load(const.path4)
    camera = Camera()
    focal_calculated = False

    for t, i in images:

        img = Image(i)
        res, key_points = img.get_points()
        if not res:
            print("no key points founded")
            break

        # draw circles
        for p in key_points:
            cv2.circle(img.img, (p[0], p[1]), 30, (0, 0, 255))
        # Вписываем центры в квадрат
        x, y, w, h = cv2.boundingRect(np.array(key_points))
        cv2.rectangle(img.img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # # find line by manual
        # x = key_points[0][0] - key_points[1][0]
        # y = key_points[0][1] - key_points[1][1]
        # length = math.sqrt(x ** 2 + y ** 2)
        if not focal_calculated:
            camera.calculate_f_length(w, const.KNOWN_SIZE, const.TEST_HIGHT)
            focal_calculated = True

        distance = camera.calculate_distance_from_camera(200, w)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img.img, t, (30, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img.img, "%.0f" % distance + "mm", (30, 80), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img.img, "rectangle %dx%d" % (h, w) + "mm", (30, 120), font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("image", img.img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
