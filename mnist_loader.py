# mnist_loader.py
import numpy as np
import struct

def load_mnist(images_path, labels_path):
    with open(labels_path, 'rb') as lbpath:
        magic, num = struct.unpack('>II', lbpath.read(8))
        y_train = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        x_train = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(y_train), 28, 28)

    x_train = x_train.astype('float32') / 255.0  # Normalizar entre 0 y 1
    return x_train, y_train
