import yaml
from utils.license_plate_general import load_model, preprocess_image, get_boxes, visualize_img
from utils.general import non_max_suppression
from utils.general import check_yaml

class Detector:
    def __init__(self, config_path):
        config = check_yaml(config_path)
        with open(config, errors='ignore') as f:
            hyp = yaml.safe_load(f)
            self.model, self.name = load_model(hyp['model_path'])

            imgsz = hyp['imgsz']
            if isinstance(imgsz, int):
                imgsz = (imgsz, imgsz)
            self.imgsz = imgsz
            
            self.conf_thres = hyp['conf_thres']
            self.iou_thres = hyp['iou_thres']
            self.max_det = hyp['max_det']
            self.agnostic_nms = hyp['agnostic_nms']
            self.class_name = hyp['class_name']
            self.color_map = hyp['color_map']
            self.device = hyp['device']
            self.line_thickness = hyp['line_thickness']
            self.line_size = hyp['line_size']
            self.hide_confidence = hyp['line_size']

    def detect(self, frame):
        new_frame = preprocess_image(frame, self.imgsz, self.device)
        prediction = self.model(new_frame, augment = False, visualize = False)[0]
        post_process_prediction = non_max_suppression(prediction, self.conf_thres, self.iou_thres, None, self.agnostic_nms, self.max_det)
        boxes = get_boxes(pred = post_process_prediction, pred_size = self.imgsz, src_size = frame.shape, conf_thres = self.conf_thres, cls = None) # Get all classes from
        return boxes

    def visualize(self, frame, boxes, using_tracking = False):
        visualized_img = frame.copy()
        for box in boxes:
            visualized_img = visualize_img(img_src = visualized_img, box = box, line_thickness = self.line_thickness, line_size = self.line_size, \
                                            color_map = self.color_map, class_name = self.class_name, \
                                            hide_confidence = self.hide_confidence, using_tracking = using_tracking)

        return visualized_img









        
        