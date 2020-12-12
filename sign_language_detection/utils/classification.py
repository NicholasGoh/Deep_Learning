import tensorflow as tf
from tensorflow.keras import layers, callbacks, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import pandas as pd
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import cv2, os, telegram_send, tqdm
from .visualization import save
plt.style.use('seaborn-white')

class Classifier:
    def __init__(self):
        self.categories = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.dict_map = {i: v for i, v in enumerate(self.categories)}
        self.num_classes = len(self.categories)
        self.img_size = 28
        self.batch_size = 128
        self.train_generator, self.test_generator = None, None
        self.step_size_train, self.step_size_valid = None, None
        self.train_path, self.test_path = None, None
        self.images, self.labels = None, None
        self.classifier = None
        self.history = None
        self.grad_cam_names = None
        self.save_folder = None

    def save_mnist(self, train_csv, test_csv, save=True):
        for folder, csv_path in [['train', train_csv], ['test', test_csv]]:
            save_dir = '/'.join(csv_path.split('/')[:-1]) + f'/{folder}/%s'
            self.train_path = save_dir[:-3] if folder == 'train' else self.train_path
            self.test_path = save_dir[:-3] if folder == 'test' else self.test_path

            for a in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                os.makedirs(save_dir % a, exist_ok=True)
        
            df = pd.read_csv(csv_path)
            counter = 0
            if save:
                for i in tqdm.tqdm(range(len(df))):
                    data = df.iloc[i]
                    image = np.resize(np.array(data[1:]), (self.img_size, self.img_size))
                    label = data[0]
                    save_path = save_dir % self.dict_map[label] + f'/{counter}.jpg'
                    cv2.imwrite(save_path, image)
                    counter += 1

    def generate_data(self, batch_size, figsize=(15,15), fontsize=16):
        self.batch_size = batch_size

        test_datagen = ImageDataGenerator(
                rescale=1/255.)
        train_datagen = ImageDataGenerator(
                rescale=1/255.,
                brightness_range=[.9, 1.],
                rotation_range=5,
                width_shift_range=0.1,
                height_shift_range=0.1)
        self.train_generator = train_datagen.flow_from_directory(
                self.train_path,
                shuffle=True,
                target_size=(self.img_size, self.img_size),
                color_mode='grayscale',
                batch_size=batch_size,
                seed=0,
                class_mode="categorical")
        self.test_generator = test_datagen.flow_from_directory(
                self.test_path,
                shuffle=True,
                target_size=(self.img_size, self.img_size),
                color_mode='grayscale',
                batch_size=batch_size,
                seed=0,
                class_mode="categorical")

        _, axes = plt.subplots(6, 6, figsize=figsize)
        for i, category in enumerate(self.categories[:6]):
            path = self.train_path + '/' + category
            images = os.listdir(path)
            for j in range(6):
                image = cv2.imread(path + '/' + images[j])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                axes[i, j].imshow(image/255.)
                axes[i, j].set(xticks=[], yticks=[])
                axes[i, j].set_title(category, color = 'tomato').set_size(fontsize)
        self.step_size_train=int(self.train_generator.n // self.train_generator.batch_size)
        self.step_size_valid=int(self.test_generator.n // self.test_generator.batch_size)

    def notify(self, fig):
        fig.savefig('tmp.jpg')
        with open('tmp.jpg', 'rb') as f:
            telegram_send.send(images=[f])
        os.remove('tmp.jpg')

    def plot_accuracy(self, history):
        f, axes = plt.subplots(1, 2, figsize=(12, 4))
        accuracy = history.history['accuracy']
        loss = history.history['loss']
        val_accuracy = history.history['val_accuracy']
        val_loss = history.history['val_loss']
        print('Training accuracy: {:.{}f}'.format(np.max(accuracy), 3))
        print('Training loss: {:.{}f}'.format(np.max(loss), 3))
        print('Validation accuracy: {:.{}f}'.format(np.max(val_accuracy), 3))
        print('Validation loss: {:.{}f}'.format(np.max(val_loss), 3))
        axes[0].plot(history.history['accuracy'])
        axes[0].plot(history.history['val_accuracy'])
        axes[0].set_title('Model accuracy')
        axes[0].set(ylabel = 'accuracy', xlabel = 'Epoch')
        axes[0].legend(['Train', 'Test'], loc='upper left')
        axes[1].plot(history.history['loss'])
        axes[1].plot(history.history['val_loss'])
        axes[1].set_title('Model loss')
        axes[1].set(ylabel = 'Loss', xlabel = 'Epoch')
        axes[1].legend(['Train', 'Test'], loc='upper left')
        return f

    def train(self,
              lr=None,
              optimizer=None,
              epochs=None,
              decay_lr=False,
              save_folder=None,
              notification = False):

        self.save_folder = save_folder
        K.clear_session()
        inputs = layers.Input(shape=(self.img_size, self.img_size, 1))
        x = layers.Conv2D(16, 3, padding = 'same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(32, 3, padding = 'same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Flatten()(x)
        # x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(self.num_classes, activation='softmax')(x)
        self.classifier = Model(inputs, x)

        self.classifier.compile(optimizer=optimizer(lr=lr),
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])

        def lr_decay(epoch):
            alpha, decay = 1, 1
            return lr / (alpha + decay * epoch)
        callback_learning_rate = callbacks.LearningRateScheduler(lr_decay, verbose=True)
        plot_losses = PlotLosses()
        callback_is_nan = callbacks.TerminateOnNaN()
        callback_early = callbacks.EarlyStopping(monitor='loss', min_delta = .001, patience = 10)

        all_callbacks = [plot_losses, callback_is_nan, callback_early]
        all_callbacks += [callback_learning_rate] if decay_lr else []

        self.history = self.classifier.fit(
                  x=self.train_generator,
                  epochs=epochs,
                  workers=15,
                  steps_per_epoch=self.step_size_train,
                  validation_steps=self.step_size_valid,
                  validation_data=self.test_generator,
                  callbacks=all_callbacks)

        fig = self.plot_accuracy(self.history)
        if save_folder:
            save(save_folder, 'acc_loss', fig=fig)
            save(save_folder, 'model', self.classifier)
        if notification:
            self.notify(fig)
        self.images, self.labels = self.test_generator.next()

    def _visualize_feature_maps(self, image):
        model_layers = self.classifier.layers
        # Extracts the outputs
        layer_outputs = [layer.output for layer in self.classifier.layers]
        # Creates a model that will return these outputs, given the model input
        activation_model = Model(inputs=self.classifier.inputs, outputs=layer_outputs)
        # get activations
        activations = activation_model.predict(image)
        images_per_row = 4; count = -1
        # Displays the feature maps
        for layer, layer_activation in zip(model_layers, activations):
            if not isinstance(layer, layers.Conv2D):
                continue
            count += 1
            # show first 3 conv layers
            if count == 3:
                break
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
            scale = 2 / size
            fig = plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer.name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='gray')
        if self.save_folder:
            save(self.save_folder, 'feature_maps', fig=fig)
        return [layer.name for layer in self.classifier.layers if isinstance(layer, layers.Conv2D)]

    def visualize_feature_maps(self, index):
        image = self.images[index:index+1]
        self.grad_cam_names = self._visualize_feature_maps(image)

    def visualize_heat_maps(self, index, rows=9, figsize=(15, 15)):
        if self.grad_cam_names == None:
            self.grad_cam_names = [layer.name for layer in self.classifier.layers if isinstance(layer, layers.Conv2D)]
        f, axes = plt.subplots(rows, 3, figsize=figsize)
        for i in range(rows):
            image = self.images[i+index:i+index+1]
            preds = self.classifier(image)
            idx = np.argmax(preds[0])
            # initialize our gradient class activation map and build the heatmap
            cam = GradCAM(self.classifier, idx, self.grad_cam_names[-1])
            heatmap = cam.compute_heatmap(image)
            (heatmap, output) = cam.overlay_heatmap(heatmap, image, self.img_size, alpha=0.4)
            description = 'image\ntrue: {} pred: {} confidence: {:.3f}'.format\
                    (self.dict_map[np.argmax(self.labels[i+index])], self.dict_map[idx], preds[0][idx])
            
            axes[i, 0].imshow(image.reshape(self.img_size, self.img_size))
            axes[i, 1].imshow(heatmap.reshape(self.img_size, self.img_size))
            axes[i, 2].imshow(output.reshape(self.img_size, self.img_size))
            axes[i, 0].set_title(description).set_size(12)
            axes[i, 1].set_title('heatmap')
            axes[i, 2].set_title('output')
            axes[i, i%3].axis('off')
        plt.tight_layout()
        if self.save_folder:
            save(self.save_folder, 'heat_maps', fig=f)

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
        # return the resulting heatmap to the calling function
        return 1-heatmap
    def overlay_heatmap(self, heatmap, image, img_size, alpha=0.2):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        output = image*(1-alpha) + heatmap.reshape(1, img_size, img_size, 1)*alpha
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        output = tf.clip_by_value(output, 0, 1).numpy()
        return (heatmap, output)

class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        display.clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()
