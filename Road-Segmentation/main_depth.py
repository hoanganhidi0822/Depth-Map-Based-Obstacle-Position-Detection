import torch
import cv2
import time
import numpy as np
from midas.model_loader import default_models, load_model
import torch.nn.functional as F
from PIL import Image
from models.pidnet import get_pred_model
from utils_seg import collision
from utils_seg import get_contour as gcnt


# SegmentPredictor class for segmentation
class SegmentPredictor:
    def __init__(self, weight, device):
        self.weight = weight
        self.device = device

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.color_map = [
            (0, 0, 0),      # background
            (31, 120, 180), # road
            (227, 26, 28),  # person 
            (106, 61, 154), # car
        ]

        get_model = get_pred_model('s', num_classes=4)
        self.model = self.load_model(get_model, weight=weight).eval()

        self.model = self.model.to(self.device)

    def prepare_input(self, image):
        image = cv2.resize(image, (640, 480))
        # normalize
        image = image.astype(np.float32) / 255.0
        image -= self.mean
        image /= self.std
        image = torch.Tensor(image).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        return image

    def load_model(self, model, weight):
        weight_dict = torch.load(weight, map_location=torch.device(self.device))
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


# Depth estimation functions
def process(device, model, model_type, image, input_size, target_size, optimize):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?

    Returns:
        the prediction
    """
    sample = torch.from_numpy(image).to(device).unsqueeze(0)

    if optimize and device.type == "cuda":
        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()

    prediction = model.forward(sample)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .detach()  # Disable gradient calculation
        .cpu()
        .numpy()
    )

    return prediction





def create_side_by_side(image, depth, grayscale):
    # Normalize depth map
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)

    # Center coordinates and depth value
    center_x, center_y = 320, 100
    center_depth = 255 - normalized_depth[center_y, center_x]
    center_depth_meters = np.median(center_depth)

    scale_factor = 60
    distance = center_depth_meters / scale_factor

    fx = 500  # Focal length x
    fy = 500  # Focal length y
    cx = depth.shape[1] / 2
    cy = depth.shape[0] / 2

    x_offset = ((center_x - cx) / fx) * distance
    y_offset = ((center_y - cy) / fy) * distance

    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2)
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    

    if image is None:
        return right_side
    else:
        return np.concatenate((image, right_side), axis=1)

def calculate_depth_and_offset(image, depth_map, bounding_boxes, frame_width, frame_height):
    scale_factor = 15  # Hệ số tỷ lệ
    fx, fy = 265, 265  # Tiêu cự camera
    cx, cy = frame_width / 2, frame_height / 2  # Điểm trung tâm camera
    
    for bbox_coords in bounding_boxes:
        xmin, ymin, xmax, ymax = bbox_coords
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        
        # Vẽ bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        
        # # Trích xuất giá trị độ sâu
        # depth_values_bbox = depth_map[ymin:ymax, xmin:xmax]
        # depth_value = np.median(depth_values_bbox)
        
        # Tính khoảng cách
        # distance = depth_value * scale_factor
        
        # # Tính tọa độ trung tâm bounding box
        # center_x = (xmin + xmax) / 2
        # center_y = (ymin + ymax) / 2
        
        # # Tính offset trong hệ tọa độ camera
        # x_offset = ((center_x - cx) / fx) * distance
        # y_offset = ((center_y - cy) / fy) * distance
        
        # # Hiển thị thông tin độ sâu và offset
        # offset_text = f"Distance: {distance:.2f} m, X: {x_offset:.2f} m, Y: {y_offset:.2f} m"
        # cv2.putText(image, offset_text, (xmin, ymax + 15), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return image

# Main video processing
def process_video():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    segment_predictor = SegmentPredictor('./weights/model_PID_s_3_class.pt', device)
    model_weights = default_models["dpt_large_384"]
    depth_model, transform, net_w, net_h = load_model(device, model_weights, "dpt_large_384", True, None, False)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Camera intrinsics
    fx, fy = 265, 265
    cx, cy = 293, 245

    fps = 0
    time_start = time.time()

    camera_matrix = np.loadtxt('camera_param/camera_matrix.txt')
    dist_coeffs = np.loadtxt('camera_param/distortion_coefficients.txt')
    
    while True:
        ret, frame = cap.read()
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        
        if not ret:
            break

        # Segmentation
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        seg_output = segment_predictor(img_rgb)
        seg_img, rgb_img, img_cb = segment_predictor.visualize(seg_output, img_rgb)

        bbs = gcnt.get_bbox(seg_img)
        bb = bbs.get_bbox()
        image_bb = gcnt.visualize(frame, bb)
        # print(bb)
        # Depth estimation
        original_image_rgb = np.flip(frame, 2)
        image = transform({"image": img_rgb / 255})["image"]
        depth_map = process(device, depth_model, "dpt_large_384", image, (net_w, net_h), frame.shape[1::-1], True)
        depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        # Invert the depth map
        depth_map = 1.0 - depth_map_normalized
        
        # Camera parameters
        camera_height = 2.0  # Độ cao của camera (m)
        camera_angle = 25.0  # Góc nghiêng của camera xuống mặt đất (độ)
        camera_angle_rad = np.radians(camera_angle)  # Chuyển góc sang radian

        # for b in bb:
        for (x, y, w, h) in bb[1]:
            xmin, ymin, xmax, ymax = int(x), int(y), int(x + w), int(y + h)
            
            # Get the depth values within the bounding box
            depth_values_bbox = depth_map[int(ymin):int(ymax), int(xmin):int(xmax)]
            depth_value = np.median(depth_values_bbox)

            # # Estimate distance in meters using the depth value (adjust the scale factor as needed)
            scale_factor = 6  # Adjust this based on the scale of your depth map
            distance = depth_value * scale_factor
            # distance = distance//10000
            # distance = depth_value
            
            if depth_values_bbox.size == 0:
                continue  # Skip if no valid depth data

            # Calculate offsets
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            x_offset_camera = ((center_x - cx) / fx) * distance
            y_offset_camera = ((center_y - cy) / fy) * distance

            # Chuyển đổi tọa độ từ camera sang mặt đất
            # Sử dụng góc nghiêng và độ cao để tính toán tọa độ thực tế
            distance_ground = np.sqrt(distance**2 - camera_height**2)
            x_offset_ground = x_offset_camera
            y_offset_ground = (distance_ground * np.cos(camera_angle_rad) - camera_height * np.sin(camera_angle_rad))

            # Display offset and depth
            depth_text = f"Depth (ground): {distance_ground:.2f} m"
            offset_text = f"X: {x_offset_ground:.2f} m, Y: {y_offset_ground:.2f} m"
            text_y_start = ymin - 15 if ymin - 15 > 10 else ymin + 15
            cv2.putText(frame, depth_text, (xmin, text_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)
            print(distance_ground) 
              
                # cv2.putText(frame, offset_text, (xmin, text_y_start + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        # Visualization
        side_by_side = create_side_by_side(image_bb, depth_map, False)
        cv2.imshow("Segmentation & Depth Map", side_by_side / 255)

        # FPS Calculation
        fps = (1 - 0.1) * fps + 0.1 * 1 / (time.time() - time_start)
        time_start = time.time()
        # print(fps)
        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    process_video()
