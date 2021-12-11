import cv2
import os
import time
import argparse

from utils.detector import Detector
from utils.license_plate_general import get_loader, perspective_transform, \
                                        drop_cls, crop_box, make_dir, \
                                        convert_boxes_to_ratio, enhance_contrast, get_value
from utils.sort import Sort


def run(license_plate_config, character_config, save_dir, visualize, save_img): 
    license_plate_detector = Detector(license_plate_config)
    character_detector = Detector(character_config)

    license_plate_tracker = Sort(license_plate_config)

    dataset, video_name = get_loader(license_plate_config)
    print("Load video: ", video_name)

    if save_img:
        save_dir = os.path.join(save_dir, video_name)
        make_dir(save_dir)

    for _, image, _, count in dataset:
        start_time = time.time()
        boxes = license_plate_detector.detect(image)

        boxes = drop_cls(boxes)
        tracking_boxes = license_plate_tracker.update(boxes)

        for tracking_box in tracking_boxes:
            raw_plate_img = crop_box(image, tracking_box, 5)
            _, enhanced_img = enhance_contrast(raw_plate_img)
            plate_img = perspective_transform(enhanced_img) 
            
            character_boxes = character_detector.detect(plate_img)
            
            new_boxes = convert_boxes_to_ratio(character_boxes, plate_img.shape)
            result = get_value(new_boxes)

            plate_img = character_detector.visualize(plate_img, character_boxes)

            if save_img:
                id = tracking_box[4]
                track_dir = os.path.join(save_dir, str(int(id)))        
                make_dir(track_dir)

                img_path = os.path.join(track_dir, str(count) + "_" + result + ".png")
                raw_img_path = os.path.join(track_dir, str(count) + "_raw.png")
                enhanced_img_path = os.path.join(track_dir, str(count) + "_enhanced.png")
                cv2.imwrite(img_path, plate_img)
                cv2.imwrite(enhanced_img_path, enhanced_img)
                cv2.imwrite(raw_img_path, raw_plate_img)
             
        if visualize:
            result_img = license_plate_detector.visualize(image, tracking_boxes, using_tracking = True)
            result_img = cv2.resize(result_img, (1280, 720))
            cv2.imshow("Result", result_img)
            cv2.waitKey(1)

        print("FPS: ", 1/(time.time() - start_time))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--license_plate_config', type=str, default= r'config\license_plate_config.yaml', help ='path to config file')
    parser.add_argument('--character_config', type=str, default= r'config\character_config.yaml', help ='path to config file')
    parser.add_argument('--save_dir', type=str, default= r'output', help='save result in this dir')
    parser.add_argument('--visualize', action='store_true', help='visualize video image')
    parser.add_argument('--save_img', action='store_true', help='save image')
  
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
