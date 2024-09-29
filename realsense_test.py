from PIL import Image
from lang_sam import LangSAM
import numpy as np
import socket
import gc
import torch
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs

if __name__ == '__main__':
    try:
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)

        model = LangSAM()
        while True:
            # This call waits until a new coherent set of frames is available on a device
            # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
            print('waiting...')
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color: continue

            img = np.asanyarray(color.get_data())
            # print(img.shape)
            # # exit(0)
            text_prompt = 'black bottle. transparent bottle. orange can. black can. white can. brown box.'
            pil_img = Image.fromarray(img, mode="RGB")
            masks, boxes, phrases, logits = model.predict(pil_img, text_prompt)
            print(boxes)
            print(phrases)
            print(masks.shape)
            # plt.imshow(img)
            # plt.show()
            cv2.imshow('image', img)
            for i, m in enumerate(masks):
                # plt.imshow(m)
                # plt.show()
                cv2.imshow(f'mask {i}', m.detach().cpu().numpy().astype(float))
            cv2.waitKey(0)
            # if len(masks.shape) == 2:
            #     masks = np.expand_dims(masks, axis=0)
            #     print('type 2', masks.shape)
            # elif len(masks.shape) == 1:
            #     masks = np.zeros((1, img.shape[0], img.shape[1]))
            #     print('type 1')
            # else:
            #     masks = masks.detach().cpu().numpy()
            print('all ok')
        exit(0)

    except Exception as e:
        print(e)
        pass