"""
Credits: mostly copy-pasted from these repo:
https://github.com/facebookresearch/detr/blob/master/datasets/coco.py
https://github.com/facebookresearch/detr/blob/master/datasets/transforms.py
https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
"""

"""
We use the same bbox annotation format of COCO dataset.

bbox = [x, y, w, h] 


:(0,0)                                         (width, 0):
:                                                        :
:              (x,y)-----------------+                   :
:                |                   |                   :
:                h     (xc, yc)      |                   :
:                |                   |                   :
:                +---------w---------+                   :
:                                                        :
:                                                        :
:(0, height)                              (width, height):


Where:
- x is the point (corner) in the X axis (from left to right) closer to the left border, in pixels.
- y is the point (corner) in the Y axis (from top to bottom) closer to the top border, in pixels.
- w is the width of the bounding box (rectangle) in pixels.
- h is the height of the bounding box (rectangle) in pixels.
- height is the height of the image in pixels.
- width is the width of the image in pixels.
"""

import torchvision
from torchvision.ops.boxes import box_area
import torch
import numpy as np

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def bboxes_overlaps(bbox1, bbox2):
    """
    Args:
        bbox1 (list): A bounding box with format x, y, w, h
        bbox2 (list):A bounding box with format x, y, w, h
    Returns:
        overlap (boolean): a boolean value which tells if the 2 bounding boxes overlaps or not
    """
    
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    iou = iou_metric(bbox1, bbox2)
    
    if iou>0:
        return True
    else:
        return False
    
def iou_metric(bbox1, bbox2):
    """
    Args:
        bbox1 (list): A bounding box with format x, y, w, h
        bbox2 (list):A bounding box with format x, y, w, h
    Returns:
        overlap (boolean): a boolean value which tells if the 2 bounding boxes overlaps or not
    """
    
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    iou = torchvision.ops.box_iou(torch.tensor(np.expand_dims([x1, y1, x1+w1, y1+h1], axis = 0)),
                           torch.tensor(np.expand_dims([x2, y2, x2+w2, y2+h2], axis = 0))) #Both sets of boxes are expected to be in (x1, y1, x2, y2) format in box_iou 
    return iou.squeeze()

def from_original_to_relative_boxes(boxes, size):
    """
    Args:
        boxes (list): A list of bboxes. Each element in the list is a bbox (list) with the form -> bbox = [x, y, w, h] 
        size (tuple): tuple composed of (width, height)

    Returns:
        boxes (list): A list of bboxes  with the form -> bbox = [xc, yc, w, h]/[width, height, width, height]
    """
    
    #first, we pass from [x, y, w, h] to [x, y, x+w, y+h]

    width, height = size
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4) 
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=width) #this shouldn t change boxes I think
    boxes[:, 1::2].clamp_(min=0, max=height) #this shouldn t change boxes I think

    #now boxes is in the form [x, y, x+w, y+h] where (x,y) is the green point and (x+w, y+h) is the yellow one
 
    boxes = box_xyxy_to_cxcywh(boxes)
    boxes = boxes / torch.tensor([width, height, width, height], dtype=torch.float32) # relative coords -> [xc, yc, w, h]/[width, height, width, height]

    return boxes.tolist() 

def box_xyxy_to_cxcywh(x):
    """
    Args:
        x (tensor): A tensor of bboxes in the form [x, y, x+w, y+h] where (x,y) is the green point and (x+w, y+h) is the yellow one

    Returns:
        boxes (tensor): A tensor of bboxes in the form [(x + x+w)/2, (y + y+h)/2, w, h] = [xc, yc, w, h] where (xc, yc) is the red star

    Copy-paste from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def from_relative_to_original_boxes(relative_boxes, size):
    """
    Args:
        boxes (list): A list of relative bboxes with the form -> bbox = [xc, yc, w, h]/[width, height, width, height]
        size (tuple): tuple composed of (width, height) of the original image

    Returns:
        boxes (list):  A list of bboxes. Each element in the list is a bbox (list) with the form -> bbox = [x, y, h, w] 
    """
    width, height = size
    boxes = box_cxcywh_to_xyxy(torch.as_tensor(relative_boxes))
    boxes = boxes * torch.tensor([width, height, width, height], dtype=torch.float32, device = boxes.device)
    boxes[:, 2:] -= boxes[:, :2]
    return boxes.tolist()

def box_cxcywh_to_xyxy(x):
    """
    Args:
        boxes (tensor): A tensor of bboxes in the form [(x + x+w)/2, (y + y+h)/2, w, h] = [xc, yc, w, h] where (xc, yc) is the red star

    Returns:
        x (tensor): A tensor of bboxes in the form [x, y, x+w, y+h] where (x,y) is the green point and (x+w, y+h) is the yellow one
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)