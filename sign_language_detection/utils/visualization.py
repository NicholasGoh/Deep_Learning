import os, scipy.io, cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers, callbacks, Model
import matplotlib.pyplot as plt

def mat(path, grid=3, skip=None, figsize=(12, 12)):
    f, axes = plt.subplots(grid, grid, figsize=figsize)
    files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith('.mat')]
    files = files[::skip] if skip else files

    for i, file in enumerate(files[:grid*grid]):
        data = scipy.io.loadmat(file, squeeze_me=True, simplify_cells=True)
        boxes = data['boxes']
        img = cv2.imread(file.replace('annotations', 'images').replace('.mat', '.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for box in boxes:
            try:
                keys = list('abcd')
                values = [box[x].astype(int) for x in keys]
                allX = [x[1] for x in values]; allY = [x[0] for x in values]
            except Exception as e:
                continue

            topleft = (min(allX), max(allY))
            bottomright = (max(allX), min(allY))
            cv2.rectangle(img, topleft, bottomright, (255, 0, 0), thickness=2)
        axes[i//grid, i%grid].imshow(img)
        axes[i//grid, i%grid].axis('off')
    plt.tight_layout()

def yolo(path, grid=3, skip=None, figsize=(12, 12)):
    f, axes = plt.subplots(grid, grid, figsize=figsize)
    files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith('.jpg')]
    files = files[::skip] if skip else files

    for i, file in enumerate(files[:grid*grid]):
        with open(file.replace('.jpg', '.txt'), 'r') as f:
            boxes = [x.split(' ') for x in f.readlines()]
        img = cv2.imread(file.replace('.txt', '.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        for box in boxes:
            try:
                _, x, y, w, h = [float(x) for x in box]
                x *= width; w *= width
                y *= height; h *= height

            except Exception as e:
                print('Unknown yolo error: %s %s' % (e, file))
                continue

            topleft = (int(x - w / 2), int(y + h / 2))
            bottomright = (int(x + w / 2), int(y - h / 2))
            cv2.rectangle(img, topleft, bottomright, (255, 0, 0), thickness=2)
        axes[i//grid, i%grid].imshow(img)
        axes[i//grid, i%grid].axis('off')
    plt.tight_layout()

def save(save_folder, folder, model=None, fig=None):
    save_folder = os.path.join(save_folder, folder)
    os.makedirs(save_folder, exist_ok=True)
    counter = 0
    if fig:
        save_path = save_folder + f'/{folder}_%d.jpg'
        while os.path.isfile(save_path % counter):
            counter += 1
        fig.savefig(save_path % counter)
    elif model:
        save_path = save_folder + f'/{folder}_%d.h5'
        while os.path.isfile(save_path % counter):
            counter += 1
        model.save(save_path % counter)
