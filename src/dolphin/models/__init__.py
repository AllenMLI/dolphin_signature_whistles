from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import Xception

from dolphin.models.basic_cnn import three_layer_cnn

MODELS = {
    'mobilenetv2': MobileNetV2,
    'resnet50': ResNet50,
    'xception': Xception,
    'threelayercnn': three_layer_cnn
}