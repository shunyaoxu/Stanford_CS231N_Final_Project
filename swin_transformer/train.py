#%% Setup
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import numpy as np
import tensorflow as tf
import time
import argparse
import tensorflow
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import MaxPooling1D, Conv1D, GlobalAveragePooling1D, Reshape
from tensorflow.keras.layers import TimeDistributed, GlobalAveragePooling2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Input, Dropout, Conv2D, BatchNormalization, MaxPool2D, Activation
from tensorflow.keras.layers import TimeDistributed as td

from keras import backend as K
from keras.models import Model
from keras import optimizers, applications
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.utils import to_categorical
from model import SwinTransformer

tensorflow.config.list_physical_devices('GPU')

parser = argparse.ArgumentParser(description='SWIN')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--decay_step', type=int, default=100,
                    help='number of step to for one decay (default: 100)')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='learning rate (default: 0.000001)')
parser.add_argument('--decay_rate', type=float, default=0.9,
                    help='learning rate decay (default: 0.9)')
args = parser.parse_args()

#%% Load Data
train_dir = '/home/users/shunyaox/dataset/output_balanced_augmented/train'
val_dir = '/home/users/shunyaox/dataset/output_balanced_augmented/val'
test_dir = '/home/users/shunyaox/dataset/output_balanced_augmented/test'
train_datagen=ImageDataGenerator(rescale=1./255, 
                                 horizontal_flip=False,
                                 vertical_flip=False)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=args.batch_size,
    class_mode="categorical",
    ) 

valid_generator=train_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=args.batch_size,
    class_mode="categorical",    
    )

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(  
        test_dir,
        target_size=(224, 224),
        batch_size=1,
        shuffle=False,
        class_mode="categorical")

#%% Check Accuracy
model = tf.keras.Sequential([
  keras.Input(shape=(224,224,3)),
  SwinTransformer('swin_large_224', num_classes=5, include_top=True, pretrained=True),
  keras.layers.Dense(5, activation='softmax')
])
#model = SwinTransformer('swin_tiny_224', num_classes=5, include_top=True, pretrained=False)
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=args.lr,
    decay_steps=args.decay_step,
    decay_rate=args.decay_rate,
    staircase=False,
    name=None
)
adam = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit_generator(generator=train_generator,
                              #steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator,
                              #validation_steps=STEP_SIZE_VALID,
                              epochs=args.num_epoch,
                              #class_weight=class_weights,
                              verbose=1).history
#%% Save Model
model.save("swin_large_224.h5")

#%% Plot
# Edit the font, font size, and axes width
mpl.rcParams['font.family'] = 'DejaVu Sans' # font
plt.rcParams['font.size'] = 18         # font size
plt.rcParams['axes.linewidth'] = 2     # axes width

fig = plt.figure(figsize=(8, 5))
ax = fig.add_axes([0, 0, 1, 1])    # Add axes object to our figure that takes up entire figure
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
ax.plot(history.history["accuracy"], linewidth=2, color='b', label="Train Acc", alpha = 1)
ax.plot(history.history["val_accuracy"], linewidth=2, color='r', label="Val Acc", alpha = 1)
ax.set_ylabel('Accuracy', labelpad=10, fontsize=20)
ax.set_xlabel('Epochs', labelpad=10, fontsize=20)
ax.grid(color='g', ls = '-.', lw = 0.5)
plt.legend(loc="lower right", fontsize=20)
plt.title("swin-l")
plt.savefig('swin_l_acc.png', dpi=300, transparent=False, bbox_inches='tight')
plt.show()

# Edit the font, font size, and axes width
mpl.rcParams['font.family'] = 'DejaVu Sans' # font
plt.rcParams['font.size'] = 18         # font size
plt.rcParams['axes.linewidth'] = 2     # axes width

fig = plt.figure(figsize=(8, 5))
ax = fig.add_axes([0, 0, 1, 1])    # Add axes object to our figure that takes up entire figure
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
ax.plot(history.history["loss"], linewidth=2, color='b', label="Train Loss", alpha = 1)
ax.plot(history.history["val_loss"], linewidth=2, color='r', label="Val Loss", alpha = 1)
ax.set_ylabel('Loss', labelpad=10, fontsize=20)
ax.set_xlabel('Epochs', labelpad=10, fontsize=20)
ax.grid(color='g', ls = '-.', lw = 0.5)
plt.legend(loc="upper right", fontsize=20)
plt.title("swin-l")
plt.savefig('swin_l_acc_loss.png', dpi=300, transparent=False, bbox_inches='tight')
plt.show()

#%% Evaluate
'''
model_loaded = keras.models.load_model("swin_large_224.h5")
outputs = model_loaded.predict(data_test)
predictions = np.argmax(outputs, axis=2)

compare = (predictions == label_test)
num_correct = np.sum(np.all(compare, axis=1))
print(f"Accuracy on Test Set = {round(num_correct / predictions.shape[0] * 100, 2)}%")
'''