import os
import numpy as np
import json
import csv
from PIL import Image, ImageDraw
import cv2


def draw_rectangle(I, xy):
    img = Image.fromarray(I)
    draw = ImageDraw.Draw(img)
    draw.rectangle([xy[1], xy[0], xy[3], xy[2]])
    del draw
    return img
"""    PIL.ImageDraw.Draw.rectangle(xy, fill=None, outline=None)
    Draws a rectangle.

    Parameters:	
    xy – Four points to define the bounding box. Sequence of either [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]. The second point is just outside the drawn rectangle.
    outline – Color to use for the outline.
    fill – Color to use for the fill.
"""


def visualize_bboxes(img, bboxes, filepath=None):
    for bbox in bboxes[-10:]:
        img = draw_rectangle(img, bbox)
        img = np.asarray(img)
    if filepath is not None:
        print(filepath)
        cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    #print(img.shape)
    #cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    #cv2.waitKey(0)



if __name__ == "__main__":
    save_path = '../data/hw01_preds'
    preds = json.load(open('../data/hw01_preds/preds.json'))
    for filename in preds:
        img = Image.open(os.path.join('../data/RedLights2011_Medium', filename))
        img = np.asarray(img)
        visualize_bboxes(img, preds[filename],
            os.path.join(save_path, 'pred_'+filename))
