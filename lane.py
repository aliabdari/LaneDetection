import cv2
import numpy as np
from bg_modeling import extract_bg
import argparse

global width
global height


class Lane:
    def __init__(self, m, b):
        self.m = 0
        self.b = b


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def fit_curve2(x1, y1, x2, y2, width, height):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    y_new = 0.45 * height
    x_new = (y_new - b) / m

    if y2 < y1:
        return x1, y1, x_new, y_new
    else:
        return x2, y2, x_new, y_new


def fit_curve(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b


def create_long_lines(lines, width, height):
    for line in lines:
        for x1, y1, x2, y2 in line:
            line[0, 0], line[0, 1], line[0, 2], line[0, 3] = fit_curve2(x1, y1, x2, y2, width, height)
    return lines


def dist_point_from_line(m, b, x0, y0):
    # y - m * x - b = 0
    return np.absolute(y0 - (m * x0) - b) / np.sqrt(np.power(1, 2) + np.power(m, 2))


def find_nearest_lines(lanes_equation, box):
    diffs = []
    bottom_center_x = (box[0] + box[2]) // 2
    bottom_center_y = (box[1] + box[3])
    for m, b in lanes_equation:
        diffs.append(dist_point_from_line(m, b, bottom_center_x, bottom_center_y))

    first_min = diffs.index(sorted(diffs)[0])
    second_min = diffs.index(sorted(diffs)[1])
    return [first_min, second_min]


def find_nearest_lines2(lanes_equation, box):
    first_line = -1
    second_line = -1

    center_x = box[0] + box[2] / 2
    center_y = box[1] + box[3]

    if center_y > 720:
        return [-1]

    x_lists = []
    for m, b in lanes_equation:
        x = (center_y - b) / m
        x_lists.append(x)

    for i in range(len(x_lists) - 1):
        if center_x >= x_lists[i] and center_x <= x_lists[i + 1]:
            first_line = i
            second_line = i + 1
            break

    return [(first_line + second_line + 1) // 2]


def find_nearest_lines3(lanes_equation, box):
    detection_list = []

    center_x = box[0] + box[2] / 2

    center_y = box[1]

    box_height = box[3]

    interval = 2
    if box_height > 15:
        interval = box[3] // 15

    while center_y < box[1] + box[3]:
        if center_y > 720:
            break

        first_line = -1
        second_line = -1
        x_lists = []
        for m, b in lanes_equation:
            x = (center_y - b) / m
            x_lists.append(x)

        for i in range(len(x_lists) - 1):
            if x_lists[i] <= center_x <= x_lists[i + 1]:
                first_line = i
                second_line = i + 1
                break

        if first_line != -1:
            detection_list = detection_list + [(first_line + second_line + 1) // 2]

        center_y += interval

    if not detection_list:
        return -1
    a = np.array(detection_list)
    counts = np.bincount(a)
    return int(np.argmax(counts))


def draw_lines(img, lines_equation):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for m, b in lines_equation:
        y1 = int(.45 * height)
        x1 = int((y1 - b) / m)
        y2 = int(height)
        x2 = int((y2 - b) / m)

        cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)

    cv2.circle(blank_image, (278, 720), 10, (0, 0, 255), thickness=10)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


def points_dist(point1, point2):
    return int(np.linalg.norm(point1 - point2))


def lines_dist(line1, line2, center=False):
    if center:
        if line1[0][3] > line1[0][1]:
            line1_bottom = np.array([line1[0][2], line1[0][3]])
        else:
            line1_bottom = np.array([line1[0][0], line1[0][1]])

        return int(np.linalg.norm(line1_bottom - line2))

    if line1[0][3] > line1[0][1]:
        line1_bottom = np.array([line1[0][2], line1[0][3]])
    else:
        line1_bottom = np.array([line1[0][0], line1[0][1]])

    if line2[0][3] > line2[0][1]:
        line2_bottom = np.array([line2[0][2], line2[0][3]])
    else:
        line2_bottom = np.array([line2[0][0], line2[0][1]])

    return int(np.linalg.norm(line1_bottom - line2_bottom))


def process_lines(lines, width, height):
    bottom_center_x = width // 2
    bottom_center_y = height
    flag_while = 1
    while flag_while:
        flag_for = 1
        for i in range(lines.shape[0]):
            if flag_for == 0:
                break
            for j in range(i + 1, lines.shape[0]):
                distance = lines_dist(lines[i], lines[j])
                if distance < 90:
                    # print("dist =" + str(distance))
                    dist_line1_from_center = lines_dist(lines[i], np.array([bottom_center_x, bottom_center_y]), True)
                    dist_line2_from_center = lines_dist(lines[j], np.array([bottom_center_x, bottom_center_y]), True)

                    if dist_line1_from_center < dist_line2_from_center:
                        lines = np.delete(lines, i, axis=0)
                    else:
                        lines = np.delete(lines, j, axis=0)
                    flag_for = 0
                    break
        if flag_for == 0:
            flag_while = 1
        else:
            flag_while = 0
    return lines


def process(image, height_, width_):
    global height
    height = height_
    width = width_
    print(image.shape)
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image,
                                       np.array([region_of_interest_vertices], np.int32), )
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi / 180,
                            threshold=100,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)

    lines = process_lines(lines, width, height)

    lines = create_long_lines(lines, width, height)

    return lines


def sort_lines(lines_equation):
    sorted_lines_equation = []
    distances = []
    sorted_distances = []
    for m, b in lines_equation:
        dist = dist_point_from_line(m, b, 0, height)
        distances.append(dist)
        sorted_distances.append(dist)
        print("dist = " + str(dist))

    sorted_distances.sort()

    for dist in sorted_distances:
        idx = distances.index(dist)
        print("idx = " + str(idx))
        sorted_lines_equation.append(lines_equation[idx])

    return sorted_lines_equation


def sort_lines2(lines_equation):
    sorted_lines_equation = []
    x_lists = []
    sorted_x_lists = []
    for m, b in lines_equation:
        x = ((height - b) / m)
        x_lists.append(x)
        sorted_x_lists.append(x)
        print("dist = " + str(x))

    sorted_x_lists.sort()

    for dist in sorted_x_lists:
        idx = x_lists.index(dist)
        print("idx = " + str(idx))
        sorted_lines_equation.append(lines_equation[idx])

    return sorted_lines_equation


def obtain_lanes():
    background = cv2.imread('background.jpg')
    lines = process(background)

    lines_equation = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            m, b = fit_curve(x1, y1, x2, y2)
            lines_equation.append([m, b])

    return lines_equation
