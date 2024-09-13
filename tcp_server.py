'''
    This is just adapted from the example in the readme,
    The main usage is for the built image to have the weights cached.
'''

from PIL import Image
from lang_sam import LangSAM
import numpy as np
import socket
import gc
import torch
import cv2


def send_arr_to_tcp(arr, conn):
    conn.sendall(arr.shape[0].to_bytes(4, byteorder='big') + arr.shape[1].to_bytes(4, byteorder='big') + arr.shape[2].to_bytes(4, byteorder='big'))
    conn.sendall(arr.tobytes())
    data = conn.recv(2)
    assert data.decode('utf-8') == 'ok'


def recv_arr_from_tcp(conn):
    shape = conn.recv(12)
    shape = (int.from_bytes(shape[:4], byteorder='big'), int.from_bytes(shape[4:8], byteorder='big'), int.from_bytes(shape[8:], byteorder='big'))
    bytes_recv = shape[0] * shape[1] * shape[2]
    arr = b''
    while bytes_recv > 0:
        data = conn.recv(65536)
        arr += data
        bytes_recv -= len(data)
    arr = np.frombuffer(arr, dtype=np.uint8).reshape(shape)
    conn.sendall('ok'.encode('utf-8'))
    return arr


def send_str_to_tcp(s, conn):
    conn.sendall(len(s).to_bytes(4, byteorder='big'))
    conn.sendall(s.encode('utf-8'))
    data = conn.recv(2)
    assert data.decode('utf-8') == 'ok'


def recv_str_from_tcp(conn):
    length = int.from_bytes(conn.recv(4), byteorder='big')
    s = b''
    while length > 0:
        data = conn.recv(65536)
        s += data
        length -= len(data)
    conn.sendall('ok'.encode('utf-8'))
    return s.decode('utf-8')


if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 12345))
        s.listen()
        print('initialized.')
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                model = LangSAM()
                img = recv_arr_from_tcp(conn)
                text_prompt = recv_str_from_tcp(conn)
                if len(text_prompt) == 0:
                    text_prompt = 'box'
                pil_img = Image.fromarray(img, mode="RGB")
                masks, boxes, phrases, logits = model.predict(pil_img, text_prompt)
                print(masks.shape)
                if len(masks.shape) == 2:
                    masks = np.expand_dims(masks, axis=0)
                    print('type 2', masks.shape)
                elif len(masks.shape) == 1:
                    masks = np.zeros((1, img.shape[0], img.shape[1]))
                    print('type 1')
                else:
                    masks = masks.detach().cpu().numpy()
                print('all ok')
                send_arr_to_tcp(masks.astype(np.uint8), conn)
                # model.cpu()
                del model
                gc.collect()
                torch.cuda.empty_cache()
                print('model deleted.')
