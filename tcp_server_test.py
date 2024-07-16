'''
    This is just adapted from the example in the readme,
    The main usage is for the built image to have the weights cached.
'''

import numpy as np
import socket
import cv2


def send_arr_to_tcp(arr, conn):
    print(arr.shape)
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
    img = cv2.imread('../vision_test/chocolade/photo_2.jpg')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn:
        conn.connect(('127.0.0.1', 12345))
        send_arr_to_tcp(img, conn)
        send_str_to_tcp('objects', conn)
        masks = recv_arr_from_tcp(conn)
        print(masks.shape)
        cv2.imshow('mask', img)
        cv2.waitKey(0)
        for i in range(masks.shape[0]):
            cv2.imshow('mask', masks[i] * 255)
            cv2.waitKey(0)
