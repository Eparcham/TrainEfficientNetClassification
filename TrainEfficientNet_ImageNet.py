from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
import cv2
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf

data_augmentation = Sequential([
    preprocessing.RandomHeight(0.1),  # randomly adjust the height of an image by a specific amount
    preprocessing.RandomWidth(0.1),  # randomly adjust the width of an image by a specific amount
    preprocessing.RandomZoom(0.1),  # randomly zoom into an image
    preprocessing.RandomRotation(factor=0.03),
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
    preprocessing.RandomContrast(factor=0.1),
    preprocessing.RandomFlip(),
], name="data_augmentation")

class trainClassfication:

    def __init__(self):
        self.TrainMode = True
        self.IMG_SIZE = 224
        dir_ = "F:/ILSVRC2017_CLS-LOC/"
        self.DataPathLabel = dir_ + 'ImageSets/CLS-LOC/train_cls.txt'
        self.FileRead = "imagenet_2012_label_map.txt"
        self.Name_class =[]
        with open(self.FileRead,'r') as f:
            nameFile = f.readlines()
        class_ = 0
        for name in nameFile:
            i = name.split(" ")
            self.Name_class.append([i[0],i[1][0:-1],class_])
            class_+=1

        with open(self.DataPathLabel,'r') as f:
            ImageName_index = f.readlines()

        self.Name_Img = []
        for name in ImageName_index: #[0:2000]
            i = name.split(" ")
            imNm = dir_ + "Data/CLS-LOC/train/" + i[0] + ".JPEG"
            classf = i[0].split("/")
            class_ = self.SerchName(classf[0])
            self.Name_Img.append([imNm,class_])

        np.random.shuffle(self.Name_Img)
        self.BachSize = 64
        self.batch = 0
        self.TestSpilt = 0.2
        self.TestSize = int(len(self.Name_Img)*self.TestSpilt)
        self.Name_Img_test = self.Name_Img[-self.TestSize:]
        self.Name_Img = self.Name_Img[0:-self.TestSize]
        self.TestSize = len(self.Name_Img_test)
        self.TrainSize = len(self.Name_Img)

    def SerchName(self,name):
        for i in self.Name_class:
            if i[0]==name:
                return i[-1]

    def NorMal(self,im_):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        for i in range(3):
            im_[..., i] -= mean[i]
            im_[..., i] /= std[i]
        return im_

    def build_model(self):

        inputs = layers.Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3))
        x = data_augmentation(inputs)
        # x = tf.keras.layers.Rescaling(1.0 / 255)(x)
        # x = self.NorMal(x)
        model = EfficientNetB0(include_top=True, input_tensor=x, weights="imagenet") #"imagenet"
        # Freeze the pretrained weights
        model.trainable = False
        # Compile
        # model = tf.keras.Model(inputs, model.output, name="EfficientNet")
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy","top_k_categorical_accuracy"])

        return model

    def TestModel(self):
        x = layers.Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3))
        model = EfficientNetB0(include_top=True, input_tensor=x, weights="imagenet")
        acc = 0
        for ii in range(self.TestSize):
            im_ = cv2.imread(self.Name_Img_test[ii][0])
            im_ = tf.image.resize(im_, [self.IMG_SIZE, self.IMG_SIZE], method='bilinear')
            im_ = np.array(im_)
            # im_ = model.preprocess_input(im_)
            # im_ = tf.keras.applications.efficientnet.preprocess_input(im_)
            im_ = np.expand_dims(im_, 0)
            pr = model(im_)
            pr = np.array(pr)
            pred_class = np.argmax(pr)
            true_class = self.Name_Img_test[ii][1]
            if true_class==pred_class:
                acc+=1

            print("acciter:  ", (acc / (ii+1)) * 100)
        print(40*"+")
        print("acc:  ",(acc/self.TestSize)*100)



    def get_new_image(self):
        while True:
            X_f = np.zeros([self.BachSize, self.IMG_SIZE, self.IMG_SIZE, 3], dtype=np.float32)
            Y_f = np.zeros([self.BachSize, 1000], dtype=np.float32)
            add = 0
            for ii in range(self.batch,self.BachSize+self.batch):
                im_ = cv2.imread(self.Name_Img[ii][0])
                im_ = tf.image.resize(im_, [self.IMG_SIZE, self.IMG_SIZE], method='bilinear')
                im_ = np.array(im_)

                Y = np.zeros([1,1000])
                Y[0][self.Name_Img[ii][1]] = 1

                X_f[add] = im_
                Y_f[add] = Y
                add+=1
                # print(ii)

            self.batch += self.BachSize
            if self.batch >= (self.TrainSize):
                self.batch = 0

            # print(self.batch)

            yield  (X_f,Y_f)

    def get_new_image_test(self):
        while True:
            X_f = np.zeros([self.BachSize, self.IMG_SIZE, self.IMG_SIZE, 3], dtype=np.float32)
            Y_f = np.zeros([self.BachSize, 1000], dtype=np.float32)
            add = 0
            for ii in range(self.batch_test,self.BachSize+self.batch_test):
                im_ = cv2.imread(self.Name_Img_test[ii][0])
                im_ = tf.image.resize(im_, [self.IMG_SIZE, self.IMG_SIZE], method='bilinear')
                im_ = np.array(im_)

                Y = np.zeros([1,1000])
                Y[0][self.Name_Img_test[ii][1]] = 1

                X_f[add] = im_
                Y_f[add] = Y
                add+=1

            self.batch_test += self.BachSize
            if self.batch_test >= (self.TestSize):
                self.batch_test = 0


            yield  (X_f,Y_f)

    def train_code(self):
        model = self.build_model()
        model.summary()
        gen = self.get_new_image()
        test_gen =self.get_new_image_test()
        model.fit(
            gen,
            steps_per_epoch=self.TrainSize // self.BachSize,
            epochs=100,
            initial_epoch=0,
            max_queue_size=10,
            validation_data=test_gen)

cl = trainClassfication()

if cl.Train_mode:
    cl.train_code()
else:
    cl.TestModel()

# import tensorflow as tf
# from tensorflow import keras
# from keras_cv_attention_models.attention_layers import (
#     activation_by_name,
#     batchnorm_with_activation,
#     conv2d_no_bias,
#     depthwise_conv2d_no_bias,
#     drop_block,
#     layer_norm,
#     se_module,
#     output_block,
#     MultiHeadRelativePositionalEmbedding,
#     add_pre_post_process,
# )
# from keras_cv_attention_models.download_and_load import reload_model_weights
# PRETRAINED_DICT = {"coatnet0": {"imagenet": {160: "030e0e79b0624ab6511a1213b3f5d814"}}}
# import os
# from keras_cv_attention_models import coatnet
# pretrained = os.path.expanduser('~/.keras/models/coatnet0_imagenet.h5')
# mm = coatnet.CoAtNet1(input_shape=(384, 384, 3), pretrained=pretrained)

# from keras_cv_attention_models import coatnet
# mm = coatnet.CoAtNet0()
# import tensorflow as tf
# from skimage.data import chelsea
# imm = tf.keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
# pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
# print(tf.keras.applications.imagenet_utils.decode_predictions(pred)[0])
