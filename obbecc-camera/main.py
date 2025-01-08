import cv2
import numpy as np
from camera_module import CameraModule

# Camera setup
camera = CameraModule()

# Parameters for perspective transformation
src_points = np.float32([
    [180, 120],  # Top-left corner of the region to warp
    [460, 120],  # Top-right corner
    [640, 480],  # Bottom-right corner
    [0, 480]     # Bottom-left corner
])

# Destination points for the transformed image (top-down view)
dst_points = np.float32([
    [0, 0],
    [640, 0],
    [560, 480],
    [80, 480]
])

# Compute the perspective transform matrix
M = cv2.getPerspectiveTransform(src_points, dst_points)

while True:
    # Capture images
    rgb_image = camera.get_rgb_image()
    depth_image = camera.get_depth_image()

    # Flip the RGB image for visualization
    # rgb_image = cv2.flip(rgb_image, 1)

    # Convert depth to meters
    depth_in_meters = depth_image / 1000.0

    # Normalize depth for visualization
    depth_visual = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply heat map to depth image
    heat_map = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)

    # Apply perspective transform to the depth heat map
    warped_heat_map = cv2.warpPerspective(heat_map, M, (640, 480))

    # Show images
    cv2.imshow('RGB Image', rgb_image)
    cv2.imshow('Depth Map', depth_visual)
    cv2.imshow('Heat Map', heat_map )

    # Depth at the center of the image
    center_x, center_y = depth_image.shape[1] // 2, depth_image.shape[0] // 2 + 160
    depth_at_center = depth_in_meters[center_y, center_x]*np.cos(np.deg2rad(30))
    print(f"Depth at center (meters): {depth_at_center:.2f} m")

    # Exit on 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()