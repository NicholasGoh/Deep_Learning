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

def visualize(model, image, save, scale=1.):
    model_layers = model.layers
    # Extracts the outputs
    layer_outputs = [layer.output for layer in model.layers]
    # Creates a model that will return these outputs, given the model input
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    # get activations
    activations = activation_model.predict(image)
    images_per_row = 4;
    # Displays the feature maps
    for layer, layer_activation in zip(model_layers, activations):
        if not isinstance(layer, layers.Conv2D):
            continue
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                # Post-processes the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std() + 1e-8
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, # Displays the grid
                             row * size : (row + 1) * size] = channel_image
        scale = scale / size
        fig = plt.figure(figsize=(scale * display_grid.shape[1],
                         scale * display_grid.shape[0]))
        plt.title(layer.name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='gray')
        if save:
            fig.savefig(f'/home/nic/data/results/feature_maps/{save}')
        return [layer.name for layer in model.layers if isinstance(layer, layers.Conv2D)]

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()
    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs[0]],
            outputs=[self.model.get_layer(self.layerName).output,
                self.model.output])
        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)
        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        # guidedGrads = guidedGrads[0]
        guidedGrads = grads[0]
        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = ((1 - heatmap) * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap
    def overlay_heatmap(self, heatmap, image, alpha=0.2, colormap=cv2.COLORMAP_JET):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        img_size = image.shape[1]
        heatmap = cv2.applyColorMap(heatmap, colormap).reshape(1, img_size, img_size, 3)
        output = image*255*(1-alpha) + heatmap.reshape(1, img_size, img_size, 3)*alpha
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        output = np.uint8(output)
        return (heatmap, output)

def classify(feature_extractor, classifier, imgs, names, figsize=(12, 12), save=False):
    grid = 3
    row = 2 if len(imgs) < 2 else len(imgs)
    f, axes = plt.subplots(row, grid, figsize=figsize)
    for i, image in enumerate(imgs):
        preds = classifier(feature_extractor(image))
        index = np.argmax(preds[0])
        # initialize our gradient class activation map and build the heatmap
        alphabets = list('abcdefghijklmnopqrstuvwxyz')
        dict_map = {i: v for i, v in enumerate(alphabets)}

        description = 'pred: {}\nconfidence: {:.3f}'.format(dict_map[index], preds[0][index])

        cam = GradCAM(feature_extractor, index, names[-1])
        heatmap = cam.compute_heatmap(image)
        (heatmap, output) = cam.overlay_heatmap(heatmap, image, alpha=0.3)
        
        img_size = image.shape[1]
        axes[i%row, 0].imshow(image.reshape(img_size, img_size, 3))
        axes[i%row, 1].imshow(heatmap.reshape(img_size, img_size, 3))
        axes[i%row, 2].imshow(output.reshape(img_size, img_size, 3))
        axes[i%row, 0].set_title(description).set_size(12)
        axes[i%row, 1].set_title('heatmap')
        axes[i%row, 2].set_title('output')
        axes[i%row, i%3].axis('off')
    plt.tight_layout()
    if save:
        f.savefig(f'/home/nic/data/results/heatmaps/{save}')
