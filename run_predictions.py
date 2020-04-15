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


def is_intersecting(A, B):
    # Checks if bbox A and B intersect at all
    # A fixed, B intersects top
    if A[1] >= B[1] and A[1] <= B[3]:
        # intersects topright
        if A[0] >= B[0] and A[0] <= B[2]:
            return True
        # intersects topleft
        elif A[2] >= B[0] and A[2] <= B[2]:
            return True
        else:
            return False
    # B intersects bottom
    elif A[3] >= B[1] and A[3] <= B[3]:
        # intersects bottomright
        if A[0] >= B[0] and A[0] <= B[2]:
            return True
        # intersects bottomleft
        elif A[2] >= B[0] and A[2] <= B[2]:
            return True
        else:
            return False
    else:
        return False


def run_nms(bboxes):
    # Assume bboxes' 4 coord is conf
    valid_bboxes = []
    bboxes = sorted(bboxes, key=lambda x:x[4], reverse=True)

    for bbox in bboxes:
        is_valid = True
        for valid_bbox in valid_bboxes:
            if is_intersecting(bbox, valid_bbox):
                is_valid = False
                break
        if is_valid:
            valid_bboxes.append(bbox)
    return valid_bboxes


def get_normalized_vector(mat):
    vec = mat.flatten()
    return (vec - vec.mean() )/ np.linalg.norm(vec)


def normalize_image(img, source_img=None, axis=None):
    if source_img is None:
        source_img = img
    return (
        (img - source_img.mean(axis=axis, keepdims=True)) /
        np.linalg.norm(source_img)
        #source_img.std(axis=axis, keepdims=True)
    )
    """(source_img.max(axis=axis, keepdims=True) -
        source_img.min(axis=axis, keepdims=True)"""


def extract_red(img, read_only=False):
    if read_only:
        img = img.copy()
    redishness = (img[:,:,0] - (img[:,:,1] + img[:,:,2]) / 2) / 2 + img.mean()
    # print(img.shape, np.expand_dims(redishness, -1).shape)
    img = np.concatenate((img, np.expand_dims(redishness, -1)), axis=2)
    return img


class Filter:
    def __init__(self, filepath, thres, sourcepath):
        rgb_source_I = np.asarray(Image.open(os.path.join(sourcepath)))
        self.source_I = extract_red(rgb_source_I, read_only=True)
        rgb_ref_I = np.asarray(Image.open(os.path.join(filepath)))
        self.ref_I = normalize_image(extract_red(rgb_ref_I, read_only=True))
        self.ref_vec = self.ref_I.flatten()
        print(self.ref_I.shape, self.source_I.mean())
        self.ref_rows, self.ref_cols = self.ref_I.shape[:2]
        self.thres = thres
    def get_detections(self, I):
        # run convolution
        n_rows, n_cols = I.shape[:2]
        bboxes = []
        confs = []
        print(I.mean())
        #I = normalize_image(extract_red(I))
        ind = 0
        for row in range(0, int(n_rows * 0.7) - self.ref_rows, max(self.ref_rows // 10, 1)):
            for col in range(0, n_cols - self.ref_cols, max(self.ref_cols // 10, 1)):
                ind += 1
                if ind % 3 != 0:
                    continue
                sub_mat = I[row:row+self.ref_rows,
                            col:col+self.ref_cols]
                sub_mat = extract_red(sub_mat)
                sub_vec = normalize_image(sub_mat.flatten())
                conf = np.dot(sub_vec, self.ref_vec)
                confs.append(conf)
                if conf > self.thres:
                    ind -= 1
                    bboxes.append([
                        row, col,
                        row+self.ref_rows, col+self.ref_cols,
                        conf
                    ])
        print(sorted(confs)[-10:])
        print('num boxes:', len(bboxes))
        return bboxes


def detect_red_light(I, det_filters):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bboxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    
    img = I.copy()
    (n_rows,n_cols,n_channels) = np.shape(I)
    
    bboxes = []
    for det_filter in det_filters:
        bboxes.extend(det_filter.get_detections(I))
    print('# bboxes before nms:', len(bboxes))
    bboxes = run_nms(bboxes)
    print('# bboxes after nms:', len(bboxes))
    bboxes = sorted(bboxes, key=lambda x: x[4])
    print("Best bboxes:\n", bboxes[-10:])

    # Remove confidences
    if len(bboxes) == 0:
        return bboxes
    bboxes = np.array(bboxes)[:,:4].tolist()

    for bbox in bboxes[-10:]:
        img = draw_rectangle(img, bbox)
        img = np.asarray(img)

    cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

    '''
    END YOUR CODE
    '''

    for i in range(len(bboxes)):
        assert len(bboxes[i]) == 4
    
    return bboxes

# set the path to the downloaded data: 
data_path = '../data/RedLights2011_Medium'

# set the path to load the filters csv:
filters_path = '../data/filters'
filters_info_filename = 'thresholds.csv' 

# set a path for saving predictions: 
preds_path = '../data/hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f]

# get detection filters
det_filters = []
with open(os.path.join(filters_path, filters_info_filename)) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    # next(reader)
    # next(reader)
    for row in reader:
        det_filters.append(
            Filter(
                os.path.join(filters_path, row[0]),
                float(row[1]),
                os.path.join(filters_path, row[2]),
            ))

preds = {}
for i in range(len(file_names)):
    # file_names[i] = 'RL-186.jpg'
    # file_names[i] = 'RL-011.jpg'
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I, det_filters)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
