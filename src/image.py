#! image processing
import cv2
import numpy as np
import src.image_loader as img_loader
import src.constants as const
import math
from src.camera import Camera


def resize(img, new_width):
    ratio = float(new_width) / img.shape[1]
    width = new_width
    hight = int(img.shape[0] * ratio)
    new_img = cv2.resize(img, (new_width, hight), interpolation=cv2.INTER_AREA)
    return new_img, width, hight


def draw_info(img, title, distance, rectangle,point_count, text_s=0.6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, title, (30, 30), font, text_s, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "%.0f" % distance + "mm", (30, 50), font, text_s, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "rectangle %dx%d" % (rectangle[0], rectangle[1]) + "mm",
                (30, 70), font, text_s, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "number of points: %d" % point_count, (30, 90), font, text_s, (0, 0, 255), 1, cv2.LINE_AA)


class ProcessedImage:

    def __init__(self, img, is_resize=True, new_width=1000, moment_method=True, adapt_mask=False):
        self.binary_img = None
        self.contours = None
        self.IsADAPTIVE_MASK = adapt_mask
        self.IsMOMENT_METHOD = moment_method

        if is_resize:
            self._orig_img, self.width, self.hight = resize(img, new_width)
        else:
            self.hight = img.shape[0]
            self.width = img.shape[1]
            self._orig_img = img

    def get_points(self, moment_method=True):
        self._threshold(self.IsADAPTIVE_MASK)
        if moment_method:
            return self.__get_points_by_moment()
        else:
            return self.__get_point_by_detector()

    def _threshold(self, adapt_mask=False):
        gray_img = cv2.cvtColor(self._orig_img, cv2.COLOR_BGR2GRAY)

        # warning adapt_mask is very slow
        if adapt_mask:
            GAUSSIAN_AREA = (gray_img.shape[0] + gray_img.shape[1]) // 4 + 1
            mask = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, GAUSSIAN_AREA, -100)
        else:
            ret, mask = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.binary_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    def __get_points_by_moment(self):
        res = self.__get_contours()
        if res:
            return True, self._get_centers_from_contours()
        return False, None

    def __get_contours(self):
        if self.binary_img is None:
            return False
        # Находим контуры на изображении
        image, self.contours, hierarchy = cv2.findContours(self.binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not self.contours:
            return False
        return True

    def _get_centers_from_contours(self):
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
    def __get_point_by_detector(self):
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

        p_img = ProcessedImage(i)
        res, key_points = p_img.get_points()
        if not res:
            print("no key points founded")
            break

        x, y, w, h = cv2.boundingRect(np.array(key_points))
        cv2.rectangle(p_img._orig_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if not focal_calculated:
            camera.calculate_f_length(w, const.KNOWN_SIZE, const.TEST_HIGHT)
            focal_calculated = True

        distance = camera.calculate_distance_from_camera(200, w)
        draw_info(p_img._orig_img, t, distance, (h, w),len(key_points))

        cv2.imshow("image", p_img._orig_img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
