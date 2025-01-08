import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api

class CameraModule:
    def __init__(self):
        # Initialize OpenNI
        openni2.initialize()

        # Open any connected device
        self.dev = openni2.Device.open_any()

        # Create and configure color stream
        self.color_stream = self.dev.create_color_stream()
        if self.color_stream is not None:
            self.color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=640, resolutionY=480, fps=30))
            self.color_stream.start()
        else:
            raise Exception("Failed to create color stream")

        # Create and configure depth stream
        self.depth_stream = self.dev.create_depth_stream()
        if self.depth_stream is not None:
            self.depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=640, resolutionY=480, fps=30))
            self.depth_stream.start()
        else:
            raise Exception("Failed to create depth stream")

    def get_rgb_image(self):
        # Read and process color frame
        color_frame = self.color_stream.read_frame()
        color_frame_data = color_frame.get_buffer_as_triplet()
        color_img = np.frombuffer(color_frame_data, dtype=np.uint8).reshape(480, 640, 3)

        # Convert RGB to BGR for OpenCV
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        return color_img

    def get_depth_image(self):
        # Read and process depth frame
        depth_frame = self.depth_stream.read_frame()
        depth_frame_data = depth_frame.get_buffer_as_uint16()
        depth_img = np.frombuffer(depth_frame_data, dtype=np.uint16).reshape(480, 640)
        return depth_img

    def close(self):
        # Stop streams and close OpenNI
        if self.color_stream is not None:
            self.color_stream.stop()
        if self.depth_stream is not None:
            self.depth_stream.stop()
        openni2.unload()
