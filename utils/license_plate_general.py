import torch
import numpy as np
import cv2
import yaml
import os

from models.experimental import attempt_load
from utils.general import  scale_coords, check_yaml
from utils.augmentations import letterbox
from utils.datasets import LoadImages
from utils.class_config import CHARACTER_CLASS

def load_model(path, train = False):
    model = attempt_load(path, map_location='cuda')  # load FP32 model
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if train:
        model.train()
    else:
        model.eval()
    return model, names

def get_boxes(pred = None, pred_size = None, src_size = None, conf_thres = 0.5, cls = None):
    boxes = []
    for det in pred:  
        det[:, :4] = scale_coords(pred_size, det[:, :4], src_size).round()
        det = det[:, :6].cpu().detach().numpy()
        for box in det:
            if cls != None:
                if int(box[5]) == cls and float(box[4]) > conf_thres:    
                    boxes.append(box)
            else:
                if float(box[4]) > conf_thres:    
                    boxes.append(box)
    return np.array(boxes)   

def drop_cls(boxes):          
    """
        Argument: [np.array[x1, y1, x2, y2, confidence, class],..]
        Convert box [x1, y1, x2, y2, confidence, class] to [x1, y1, x2, y2, confidence]
    """
    if boxes.shape[0] == 0:
        return np.empty((0, 5))
    else:
        return boxes[:,: 5]

def preprocess_image(original_image, size = (1280,1280), device = 'cuda'):
    image = letterbox(original_image, size, stride= 8, auto = False)[0]
    image = image.transpose((2, 0, 1))[::-1]
    image = np.ascontiguousarray(image)

    image = torch.from_numpy(image).to(device)
    image = image.float()  
    image = image / 255.0 
    if image.ndimension() == 3:
        image = image.unsqueeze(0)
    return image

def visualize_img(img_src = None, box = None, line_thickness = 3, line_size = 1, color_map = None, class_name = None, hide_confidence = False, using_tracking = True):
    thickness = line_thickness
    line_size = line_size
    font = cv2.FONT_HERSHEY_SIMPLEX

    if using_tracking:
        color = (255, 0, 0)
        display_string = "ID: " + str(box[4])

        visualize_image = cv2.putText(img_src, display_string, 
                                        (int((box[0] + box[2])/2), int((box[1] + box[3])/2)), 
                                        font, line_size, color, thickness)

        visualize_image = cv2.rectangle(visualize_image, 
                                        (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 
                                        color , thickness)

    else:
        color = color_map[int(box[5])]
        if hide_confidence:
            display_string = str(class_name[int(box[5])]) 
        else:
            display_string = "{} [{:.2f}]".format(str(class_name[int(box[5])]), float(box[4]))

        visualize_image = cv2.putText(img_src, display_string, 
                                        (int((box[0] + box[2])/2), int((box[1] + box[3])/2)), 
                                        font, line_size, color, thickness)

        visualize_image = cv2.rectangle(visualize_image, 
                                        (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 
                                        color , thickness)
    return visualize_image


def get_loader(config_path):
    config = check_yaml(config_path)
    with open(config, errors='ignore') as f:
        hyp = yaml.safe_load(f)

        source = hyp['source']
        imgsz = hyp['imgsz']
        stride = hyp['stride']
        auto = hyp['auto']

    video_name = source.split('\\')[-1].split('.')[0]
            
    return LoadImages(source, img_size= imgsz, stride= stride, auto = auto), video_name


def crop_boxes(image, boxes, cls, padding = 5):
    images = []

    for box in boxes:
        if int(box[5]) == cls:
            x1 = int(box[1] - padding) if int(box[1] - padding) > 0 else 0
            x2 = int(box[3] + padding) if int(box[3] + padding) < image.shape[0] else image.shape[0]
            y1 = int(box[0] - padding) if int(box[0] - padding) > 0 else 0
            y2 = int(box[2] + padding) if int(box[2] + padding) < image.shape[1] else image.shape[1]
            cropped_image = image[x1 : x2, y1 : y2, :]
            images.append(cropped_image)

    return images

def crop_box(image, box, padding = 5):
    x1 = int(box[1] - padding) if int(box[1] - padding) > 0 else 0
    x2 = int(box[3] + padding) if int(box[3] + padding) < image.shape[0] else image.shape[0]
    y1 = int(box[0] - padding) if int(box[0] - padding) > 0 else 0
    y2 = int(box[2] + padding) if int(box[2] + padding) < image.shape[1] else image.shape[1]

    return image[x1 : x2, y1 : y2, :]


def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_transform(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY )

    cont, _ = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    if cont == ():
        return img

    cont = sorted(cont, key= lambda cont:cv2.contourArea(cont), reverse = True)
    a = cont[0].reshape(cont[0].shape[0], cont[0].shape[2])

    rect = cv2.minAreaRect(a)


    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = np.array(box)
    rect = order_points(box)

    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
    # # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    if warped.shape[0] < 20:
        return img

    if warped.shape[1] < 20:
        return img
    return warped


def make_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


def convert_boxes_to_ratio(boxes, imgsz):
    new_boxes = []
    for box in boxes:
        x_center = (box[0] + box[2])/2
        y_center = (box[1] + box[3])/2
        height = box[3] - box[1]
    
        x_center /= imgsz[1]
        y_center /= imgsz[0]
        height /= imgsz[0]

        new_box = np.array([x_center, y_center, box[4], box[5], height])
        new_boxes.append(new_box)

    return new_boxes

def check_is_square_plate(boxes):
    return np.mean([box[-1] for box in boxes]) < 0.65

def sort_boxes_along_x(boxes):
    indice = np.argsort([box[0] for box in boxes]) 
    return np.array(boxes)[indice]


def get_character(boxes):
    result = ""
    for box in boxes:
        result += CHARACTER_CLASS[int(box[-2])]

    return result


def get_value(boxes):
    if boxes == []:
        return "Empty"
    if check_is_square_plate(boxes):
        upper_character = []
        lower_character = []
        for box in boxes:
            if box[1] < 0.5:
                upper_character.append(box)
            else:
                lower_character.append(box)
        
        sorted_upper_character = sort_boxes_along_x(upper_character)
        sorted_lower_character = sort_boxes_along_x(lower_character)
        upper_string = get_character(sorted_upper_character)
        lower_string = get_character(sorted_lower_character)

        result_string = upper_string + "-" + lower_string

    else:
        sorted_character = sort_boxes_along_x(boxes)
        result_string = get_character(sorted_character)

    return result_string


def enhance_contrast(img):
  b_img, g_img, r_img = cv2.split(img)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
  equalized_b_img = clahe.apply(b_img)
  equalized_g_img = clahe.apply(g_img)
  equalized_r_img = clahe.apply(r_img)

  return cv2.merge([equalized_b_img, equalized_g_img, equalized_r_img]),\
         cv2.merge([cv2.equalizeHist(b_img), cv2.equalizeHist(g_img), cv2.equalizeHist(r_img)])
