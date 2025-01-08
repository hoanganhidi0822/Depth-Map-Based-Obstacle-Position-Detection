import cv2
import numpy as np
import time


class get_bbox:
    def __init__(self, image):
    # Định nghĩa các màu trong color_map
        self.color_map = [
            (227, 26, 28),  # person
            (106, 61, 154), # car
        ]

        # Định nghĩa màu vẽ cho từng loại contour
        self.drawing_colors = [
            (255, 255, 255), # white for background
            (255, 0, 0),     # blue for road
            (0, 255, 0),     # green for person
            (0, 0, 255),     # red for car
        ]

        self.image = image
        self.contours_dict = self.get_contour()
        self.bbox_list = [[] for _ in range(len(self.color_map))]



    def get_contour(self):

        contours_dict = {}

        # Lặp qua từng màu trong color_map
        for i, color in enumerate(self.color_map):
            # Tạo mặt nạ dựa trên giá trị màu
            mask = cv2.inRange(self.image, np.array(color), np.array(color))
            
            # Tìm các đường viền trong mặt nạ
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contours_dict[color] = contours

        return contours_dict

    def get_bbox(self):
        for i, color in enumerate(self.color_map):
            self.bbox_list[i] = [
                cv2.boundingRect(cnt) for cnt in self.contours_dict[color]
            ]
        
        # print(self.bbox_list)
        return self.bbox_list

def visualize(image, bbox_list):
    drawing_colors = [
        (0, 255, 0),     # green for person
        (0, 0, 255),     # red for car
    ]

    for color, boxes in zip(drawing_colors, bbox_list):
        for (x, y, w, h) in boxes:
            image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
    return image


if __name__ == "__main__":
    # Đọc ảnh đầu vào
    seg_image = cv2.imread('outputs/1501_segment.png')
    rgb_image = cv2.imread('outputs/1501_rgb.png')

    seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    bbs = get_bbox(seg_image)

    bb = bbs.get_bbox()

    image_bb = visualize(rgb_image, bb) 

    # print(bb)
    image_bb = cv2.cvtColor(image_bb, cv2.COLOR_RGB2BGR)

    cv2.imshow("test", image_bb)

    cv2.waitKey(0)


