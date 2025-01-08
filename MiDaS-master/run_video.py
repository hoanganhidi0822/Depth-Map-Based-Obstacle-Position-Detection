import os
import torch
import cv2
import time
import numpy as np
from midas.model_loader import default_models, load_model


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
        .detach()  # Tắt chế độ tính gradient
        .cpu()
        .numpy()
    )

    return prediction



def create_side_by_side(image, depth, grayscale):
    """
    Take an RGB image and depth map and place them side by side. This includes a proper normalization of the depth map
    for better visibility.

    Args:
        image: the RGB image
        depth: the depth map
        grayscale: use a grayscale colormap?

    Returns:
        the image and depth map place side by side
    """
    print(depth)
    # Chuẩn hóa bản đồ độ sâu
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)  # Chuẩn hóa về khoảng [0, 255]

    # Tọa độ tâm ảnh
    center_x, center_y = 320, 100

    # Giá trị độ sâu tại tâm ảnh
    center_depth = 255 - normalized_depth[center_y, center_x]  # Giá trị độ sâu tại tâm (trực tiếp từ bản đồ độ sâu gốc)
    center_depth_meters = np.median(center_depth)   # Chuyển đổi sang mét nếu cần (giả sử giá trị gốc là mm)

    # Tính khoảng cách và offset từ tâm ảnh
    scale_factor = 60  # Hệ số tùy chỉnh
    distance = center_depth_meters / scale_factor  # Khoảng cách tính toán từ độ sâu

    # Các thông số camera
    fx = 500  # Focal length theo trục x
    fy = 500  # Focal length theo trục y
    cx = depth.shape[1] / 2  # Tâm ảnh theo trục x
    cy = depth.shape[0] / 2  # Tâm ảnh theo trục y

    x_offset = ((center_x - cx) / fx) * distance
    y_offset = ((center_y - cy) / fy) * distance

    # Tạo bản đồ độ sâu hiển thị
    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) 
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    # Hiển thị giá trị độ sâu và offset tại tâm ảnh
    cv2.putText(right_side, f"Depth: {distance:.2f} m", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(right_side, f"X Offset: {x_offset:.2f} m, Y Offset: {y_offset:.2f} m", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Đánh dấu tâm ảnh
    cv2.circle(right_side, (center_x, center_y), 5, (0, 255, 0), -1)

    if image is None:
        return right_side
    else:
        return np.concatenate((image, right_side), axis=1)




def run_camera(model_path, model_type, optimize, side, height, square, grayscale):
    """
    Run depth estimation using the camera as input.

    Args:
        model_path (str): path to the trained weights
        model_type (str): model type
        optimize (bool): optimize the model to half-floats on CUDA
        side (bool): output RGB and depth side by side
        height (int): encoder input height
        square (bool): resize input to square resolution
        grayscale (bool): use grayscale colormap
    """
    print("Initializing...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Processing video stream. Press 'ESC' to exit.")

    time_start = time.time()
    fps = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Unable to read frame from the camera.")
            break

        original_image_rgb = np.flip(frame, 2)  # in [0, 255] (flip required to get RGB)
        image = transform({"image": original_image_rgb / 255})["image"]

        # Tính toán bản đồ độ sâu
        prediction = process(device, model, model_type, image, (net_w, net_h),
                             original_image_rgb.shape[1::-1], optimize)

        original_image_bgr = np.flip(original_image_rgb, 2) if side else None

        # Hiển thị bản đồ độ sâu với giá trị tại tâm
        content = create_side_by_side(original_image_bgr, prediction, grayscale)
        cv2.imshow('MiDaS Depth Estimation - Press Escape to close window', content / 255)

        alpha = 0.1
        fps = (1 - alpha) * fps + alpha * 1 / (time.time() - time_start)  # exponential moving average
        time_start = time.time()
        print(f"\rFPS: {round(fps, 2)}", end="")

        if cv2.waitKey(1) == 27:  # Escape key
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_weights = default_models["dpt_large_384"]  # Change to the desired model type
    run_camera(
        model_path=model_weights,
        model_type="dpt_large_384",
        optimize=True,
        side=True,
        height=None,
        square=False,
        grayscale=False,
    )
