import argparse
import cv2
from lane import process, sort_lines2, draw_lines, find_nearest_lines2, fit_curve
from bg_modeling import extract_bg

parser = argparse.ArgumentParser(description='Lane detection.')
parser.add_argument("--input_image", help="Input image address.format", type=str, default='./background.jpg',
                    required=False)
parser.add_argument("--input_video", help="Input video address.format", type=str, default='./Input2.mp4')

opt = parser.parse_args()
args = vars(opt)

# background obtaining from video
background = cv2.imread(args["input_image"])
if background is None:
    video = cv2.VideoCapture(args["input_video"])
    background = extract_bg(args["input_video"])

height = background.shape[0]
width = background.shape[1]
# background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
lines = process(background, height, width)

lines_equation = []

for line in lines:
    for x1, y1, x2, y2 in line:
        m, b = fit_curve(x1, y1, x2, y2)
        lines_equation.append([m, b])

lines = sort_lines2(lines_equation)
# lines = np.delete(lines,0,axis=0)
image_with_lines = draw_lines(background, lines)

find_nearest_lines2(lines, (100, 100, 200, 200))

while True:
    cv2.imshow('Lane', image_with_lines)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
