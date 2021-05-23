from const import *
from handles import CRNNHandle
from handles import  AngleNetHandle
from handles import DBNETHandle
from utils import draw_bbox, crop_rect, sorted_boxes, get_rotate_crop_image
from PIL import Image
import numpy as np
import cv2
import copy
import time
import traceback

class  OcrHandle(object):
    def __init__(self):
        self.text_handle = DBNETHandle(dbnet_model_path)
        self.crnn_handle = CRNNHandle(crnn_model_path)
        self.angle_handle = AngleNetHandle(angle_net_path)

    def receive_crnn(self,im, rgb, boxes_list, score_list, angle_detect_num):
        """
        crnn模型，ocr识别
        @@model,
        @@converter,
        @@im:Array
        @@text_recs:text box
        @@ifIm:是否输出box对应的img

        """
        results = []
        boxes_list = sorted_boxes(np.array(boxes_list))

        line_imgs = []
        for index, (box, score) in enumerate(zip(boxes_list[:angle_detect_num], score_list[:angle_detect_num])):
            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))
            partImg = Image.fromarray(partImg_array).convert("RGB")
            line_imgs.append(partImg)

        angle_res = self.angle_handle.predict_rbgs(line_imgs)

        count = 1
        for index, (box ,score) in enumerate(zip(boxes_list,score_list)):

            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))


            partImg = Image.fromarray(partImg_array).convert("RGB")

            if angle_res:
                partImg = partImg.rotate(180)

            try:
                if rgb:
                    simPred = self.crnn_handle.predict_rbg(partImg)  ##识别的文本
                else:
                    partImg = partImg.convert('L')
                    simPred = self.crnn_handle.predict(partImg)  ##识别的文本
            except Exception as e:
                print(traceback.format_exc())
                continue

            if simPred.strip() != '':
                results.append([tmp_box,"{}、 ".format(count)+  simPred,score])
                count += 1

        return results


    def text_predict(self,img, rgb, short_size,angle_detect_num):
        boxes_list, score_list = self.text_handle.process(np.asarray(img).astype(np.uint8),short_size=short_size)
        result = self.receive_crnn(np.array(img), rgb, boxes_list, score_list ,angle_detect_num)

        return result

