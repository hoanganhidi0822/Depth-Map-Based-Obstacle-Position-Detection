import torch
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
import torch

from models.pidnet import get_pred_model
from utils_seg import collision
import time
from utils_seg import get_contour as gcnt

class SegmentPredictor:
    def __init__(self, weight, device):
        self.weight = weight
        self.device = device

        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225] 

        self.color_map = [
            (0, 0, 0),      # backgroud
            (31, 120, 180), # road
            (227, 26, 28),  # person 
            (106, 61, 154), # car
            ]

        get_model = get_pred_model('s', num_classes=4)
        self.model = self.load_model(get_model, weight=weight).eval()

        self.model = self.model.to(self.device)

    def prepare_input(self, image):
        image = cv2.resize(image, (640, 480))
        # normolize
        image = image.astype(np.float32) / 255.0
        image -= self.mean
        image /= self.std
        image = torch.Tensor(image).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        return image
    
    def load_model(self, model, weight):
        weight_dict = torch.load(weight,  map_location=torch.device(self.device))
        model_dict = model.state_dict()
        weight_dict = {k: v for k, v in weight_dict.items() if k in model_dict}
        model_dict.update(weight_dict)
        model.load_state_dict(model_dict)
        return model
    
    def visualize(self, mask, image):

        sv_img = np.zeros_like(image).astype(np.uint8)
        for i, color in enumerate(self.color_map):
            for j in range(3):
                sv_img[:,:,j][mask==i] = self.color_map[i][j]

        img_cb = cv2.addWeighted(sv_img, 0.5, image, 0.5, 0)

        return sv_img, image, img_cb

    def __call__(self, image):
        image = self.prepare_input(image)
        pred = self.model(image)
        
        # postprocess
        pred = F.interpolate(pred, size=(480, 640), mode='bilinear', align_corners=True)
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = pred.cpu().squeeze(0).numpy()

        return pred
    
def check_type(x):
    if isinstance(x, list):
        print("x là một danh sách (list).")
    elif isinstance(x, np.ndarray):
        print("x là một mảng numpy (numpy array).")
    elif isinstance(x, torch.Tensor):
        print("x là một tensor của PyTorch.")

color_map = [
            (0, 0, 0),      # backgroud
            (31, 120, 180), # road
            (227, 26, 28),  # person 
            (106, 61, 154), # car
            ]

def ped_in_pov(bbox_list, quad):

    for boxes in bbox_list:
        for (x, y, w, h) in boxes:
            x1, y1 = x, y
            x2, y2 = x + w, y + h

            point_x = int((x1+x2)/2)
            point_y = y2

            if collision.is_point_in_quadrilateral_area_method(point_x, point_y, quad) == True:
                return True

    return False


def process_video():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = SegmentPredictor('./weights/model_PID_s_3_class.pt', device)

    idx = 0
    count = 0
    brake_count = 0
    brake = 0
    
    quad = np.array([[0, 479], [90, 200], [550, 200], [639, 479]])
    cap = cv2.VideoCapture(0)
    camera_matrix = np.loadtxt('camera_param/camera_matrix.txt')
    dist_coeffs = np.loadtxt('camera_param/distortion_coefficients.txt')
    

    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_image = cv2.undistort(frame, camera_matrix, dist_coeffs)
        img_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (640, 480))
        output = predictor(img_rgb)

        seg_img, rgb_img, img_cb = predictor.visualize(output, img_rgb)

        # print("----------------seg_img-------------------------")

        bbs = gcnt.get_bbox(seg_img)
        bb = bbs.get_bbox()
        print (bb)

        cv2.polylines(rgb_img, [quad], isClosed=True, color=(0,0,255), thickness = 3)

        is_collision = ped_in_pov(bb, quad)


        print(bb)
        image_bb = gcnt.visualize(rgb_img, bb) 

        end = time.time()

        fps = 1/(end-start)
        print(fps)
        img_cb = cv2.cvtColor(img_cb, cv2.COLOR_RGB2BGR)

        
        
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
        image_bb = cv2.cvtColor(image_bb, cv2.COLOR_RGB2BGR)

        cv2.imshow('im', seg_img)
        #cv2.imshow('regions', image)
        idx = idx + 1

        cv2.imshow('rgb', image_bb )

        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':

    process_video()
    cv2.destroyAllWindows()




