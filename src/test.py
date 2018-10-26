import time
import src.image as image
import src.image_loader as loader
from matplotlib import pyplot as plt
import cv2


def timed(f, *args, n_iter=10, ret_time="average"):
    min_t = float("inf")
    max_t = float("-inf")

    for i in range(n_iter):
        t0 = time.perf_counter()
        f(*args)
        t1 = time.perf_counter()
        min_t = min(min_t, t1 - t0)
        max_t = max(max_t, t1 - t0)

    if ret_time == "min":
        return min_t
    elif ret_time == "max":
        return max_t
    elif ret_time == "average":
        return (max_t + min_t) / 2
    else:
        return None


def compare(fs, args, ff=None):
    for f in fs:
        plt.plot(range(1, len(args) + 1),
                 [timed(f, ff(arg) if ff is not None else arg) for arg in args],
                 'o', label=f.__name__)
    plt.legend()
    plt.ylabel("time in seconds")
    plt.xlabel("arg number")
    plt.grid(True)
    plt.show()


def decor(f, flag):
    def inner(n):
        return f(n, flag)
    return inner


def main():
    path1 = "../resource/image/led/0"
    images = loader.load(path1)

    # # gaussian threshold vs plain threshold
    # gaussian_thresh = decor(image, True)
    # print(timed(gaussian_thresh, images[0]) / timed(image.led_thresh, images[0]))
    #
    #
    # # auto detector vs moments method
    # # compare([image.get_points_by_moment, image.get_point_by_detector], images, ff=image.led_thresh)


if __name__ == "__main__":
    main()
