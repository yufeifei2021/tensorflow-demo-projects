#  -*-coding:utf8 -*-

"""
Created on 2021 7 9

@author: 陈雨
"""

'''迁移学习'''
import  tensorflow as tf
from    tensorflow import keras

base_model = keras.applications.ResNet152(weights='imagenet')
base_model.summary()

'''ResNet50 101 152 models.Model建模方法'''
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D

# 继承Layer,建立resnet50 101 152卷积层模块
def conv_block(inputs, filter_num, stride=1, name=None):
    
    x = inputs
    x = Conv2D(filter_num[0], (1,1), strides=stride, padding='same', name=name+'_conv1')(x)
    x = BatchNormalization(axis=3, name=name+'_bn1')(x)
    x = Activation('relu', name=name+'_relu1')(x)

    x = Conv2D(filter_num[1], (3,3), strides=1, padding='same', name=name+'_conv2')(x)
    x = BatchNormalization(axis=3, name=name+'_bn2')(x)
    x = Activation('relu', name=name+'_relu2')(x)

    x = Conv2D(filter_num[2], (1,1), strides=1, padding='same', name=name+'_conv3')(x)
    x = BatchNormalization(axis=3, name=name+'_bn3')(x)
    
    # residual connection
    r = Conv2D(filter_num[2], (1,1), strides=stride, padding='same', name=name+'_residual')(inputs)
    x = layers.add([x, r])
    x = Activation('relu', name=name+'_relu3')(x)

    return x

def build_block (x, filter_num, blocks, stride=1, name=None):

    x = conv_block(x, filter_num, stride, name=name)

    for i in range(1, blocks):
        x = conv_block(x, filter_num, stride=1, name=name+'_block'+str(i))

    return x


# 创建resnet50 101 152
def ResNet(Netname, nb_classes):

    ResNet_Config = {'ResNet50':[3,4,6,3],
                    'ResNet101':[3,4,23,3],
                    'ResNet152':[3,8,36,3]}
    layers_dims=ResNet_Config[Netname]

    filter_block1=[64, 64, 256]
    filter_block2=[128,128,512]
    filter_block3=[256,256,1024]
    filter_block4=[512,512,2048]

    img_input = Input(shape=(224,224,3))
    # stem block 
    x = Conv2D(64, (7,7), strides=(2,2),padding='same', name='stem_conv')(img_input)
    x = BatchNormalization(axis=3, name='stem_bn')(x)
    x = Activation('relu', name='stem_relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same', name='stem_pool')(x)
    # convolution block
    x = build_block(x, filter_block1, layers_dims[0], name='conv1')
    x = build_block(x, filter_block2, layers_dims[1], stride=2, name='conv2')
    x = build_block(x, filter_block3, layers_dims[2], stride=2, name='conv3')
    x = build_block(x, filter_block4, layers_dims[3], stride=2, name='conv4')
    # top layer
    x = GlobalAveragePooling2D(name='top_layer_pool')(x)
    x = Dense(nb_classes, activation='softmax', name='fc')(x)

    model = models.Model(img_input, x, name=Netname)

    return model
    
def main():

    model = ResNet('ResNet50', 1000)
    model.summary()

if __name__=='__main__':
    main()


'''ResNet50 101 152 继承方法'''
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D

# 继承Layer,建立resnet50 101 152卷积层模块
class CellBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(CellBlock, self).__init__()

        self.conv1 = Conv2D(filter_num[0], (1,1), strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')

        self.conv2 = Conv2D(filter_num[1], (3,3), strides=1, padding='same')
        self.bn2 = BatchNormalization()
        self.relu2 = Activation('relu')

        self.conv3 = Conv2D(filter_num[2], (1,1), strides=1, padding='same')
        self.bn3 = BatchNormalization()

        self.residual = Conv2D(filter_num[2], (1,1), strides=stride, padding='same')
        
    def call (self, inputs, training=None):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        r = self.residual(inputs)

        x = layers.add([x, r])
        output = tf.nn.relu(x)

        return output

#继承Model， 创建resnet50 101 152
class ResNet(models.Model):
    def __init__(self, layers_dims, nb_classes):
        super(ResNet, self).__init__()

        self.stem = Sequential([
            Conv2D(64, (7,7), strides=(2,2),padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((3,3), strides=(2,2), padding='same')
        ]) #开始模块

        # 不同卷积层中的filter个数
        filter_block1=[64, 64, 256]
        filter_block2=[128,128,512]
        filter_block3=[256,256,1024]
        filter_block4=[512,512,2048]

        self.layer1 = self.build_cellblock(filter_block1, layers_dims[0]) 
        self.layer2 = self.build_cellblock(filter_block2, layers_dims[1], stride=2)
        self.layer3 = self.build_cellblock(filter_block3, layers_dims[2], stride=2)
        self.layer4 = self.build_cellblock(filter_block4, layers_dims[3], stride=2)

        self.avgpool = GlobalAveragePooling2D()
        self.fc = Dense(nb_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        x=self.stem(inputs)
        # print(x.shape)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        
        x=self.avgpool(x)
        x=self.fc(x)

        return x

    def build_cellblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(CellBlock(filter_num, stride)) #每层第一个block stride可能为非1

        for _ in range(1, blocks):      #每一层由多少个block组成
            res_blocks.add(CellBlock(filter_num, stride=1))

        return res_blocks


def build_ResNet(NetName, nb_classes):
    ResNet_Config = {'ResNet50':[3,4,6,3],
                    'ResNet101':[3,4,23,3],
                    'ResNet152':[3,8,36,3]}

    return ResNet(ResNet_Config[NetName], nb_classes) 

def main():
    model = build_ResNet('ResNet152', 1000)
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()

if __name__=='__main__':
    main()
    

'''ResNet18 34'''
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D

# 继承Layer,建立resnet18和34卷积层模块
class CellBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(CellBlock, self).__init__()

        self.conv1 = Conv2D(filter_num, (3,3), strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.relu = Activation('relu')

        self.conv2 = Conv2D(filter_num, (3,3), strides=1, padding='same')
        self.bn2 = BatchNormalization()

        if stride !=1:
            self.residual = Conv2D(filter_num, (1,1), strides=stride)
        else:
            self.residual = lambda x:x
        
    def call (self, inputs, training=None):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        r = self.residual(inputs)

        x = layers.add([x, r])
        output = tf.nn.relu(x)

        return output

#继承Model， 创建resnet18和34
class ResNet(models.Model):
    def __init__(self, layers_dims, nb_classes):
        super(ResNet, self).__init__()

        self.stem = Sequential([
            Conv2D(64, (7,7), strides=(2,2),padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((3,3), strides=(2,2), padding='same')
        ]) #开始模块

        self.layer1 = self.build_cellblock(64, layers_dims[0]) 
        self.layer2 = self.build_cellblock(128, layers_dims[1], stride=2)
        self.layer3 = self.build_cellblock(256, layers_dims[2], stride=2)
        self.layer4 = self.build_cellblock(512, layers_dims[3], stride=2)

        self.avgpool = GlobalAveragePooling2D()
        self.fc = Dense(nb_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        x=self.stem(inputs)
        # print(x.shape)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        
        x=self.avgpool(x)
        x=self.fc(x)

        return x

    def build_cellblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(CellBlock(filter_num, stride)) #每层第一个block stride可能为非1

        for _ in range(1, blocks):      #每一层由多少个block组成
            res_blocks.add(CellBlock(filter_num, stride=1))

        return res_blocks


def build_ResNet(NetName, nb_classes):
    ResNet_Config = {'ResNet18':[2,2,2,2], 
                    'ResNet34':[3,4,6,3]}

    return ResNet(ResNet_Config[NetName], nb_classes) 

def main():
    model = build_ResNet('ResNet18', 1000)
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()

if __name__=='__main__':
    main()
    





