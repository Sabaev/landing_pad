import glob
import cv2
import os.path


def load(images_folder, ext):
    ext = ext.lower()
    images = []
    for f in os.listdir(images_folder):
        f_ext = os.path.splitext(f)[1]
        if ext == f_ext:
            images.append(cv2.imread(images_folder + "/" + f))
    return images


def main():
    path = "../resource/image/led/0"
    images = load(path, ".bmp")
    n = 0
    for i in images:
        cv2.imshow("img number %s" % n, i)
        n += 1
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
