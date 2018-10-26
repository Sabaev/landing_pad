import math


class Camera:
    def __init__(self, focal=100):
        self.focal = focal
        self.pixel_per_meter = 0

    # f/p = distance/known_size => distance = f*known_size/p
    def calculate_distance_from_camera(self, known_size, pixel_size):
        self.pixel_per_meter = known_size/ pixel_size
        return float(self.focal) * self.pixel_per_meter

    def calculate_distance_between_objects(self, fst_p, snd_p):
        distance = (snd_p - fst_p)*self.pixel_per_meter
        return math.fabs(distance)

    def set_focal_length(self, f_length):
        self.focal = f_length

    def calculate_f_length(self, p_size, known_size, known_distance):
        self.focal = p_size * known_distance / known_size
        return self.focal
