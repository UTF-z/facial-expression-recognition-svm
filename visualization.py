import numpy as np
import pandas as pd
import os
import argparse
import errno
import dlib
import cv2
import matplotlib

from skimage.feature import hog
from skimage import exposure
from matplotlib import pyplot as plt
from matplotlib import patches
from data_loader import *

# initialization
image_height = 300
image_width = 300
window_size = 100
window_step = 6
ONE_HOT_ENCODING = False
SAVE_IMAGES = False
GET_LANDMARKS = False
GET_HOG_FEATURES = False
GET_HOG_WINDOWS_FEATURES = False
SELECTED_LABELS = []
IMAGES_PER_LABEL = [500, 100, 100]
OUTPUT_FOLDER_NAME = "fer2013_features"
VIS_FOLDER_NAME = 'fer2013_vis'

# parse arguments and initialize variables:
parser = argparse.ArgumentParser()
parser.add_argument("-j", "--jpg", default="no", help="save images as .jpg files")
parser.add_argument("-l", "--landmarks", default="yes", help="extract Dlib Face landmarks")
parser.add_argument("-ho", "--hog", default="yes", help="extract HOG features")
parser.add_argument("-hw", "--hog_windows", default="yes", help="extract HOG features from a sliding window")
parser.add_argument("-o", "--onehot", default="no", help="one hot encoding")
parser.add_argument(
    "-e",
    "--expressions",
    default="0,1,2,3,4",
    help=
    "choose the faciale expression you want to use: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral")
args = parser.parse_known_args()[0]
expr_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
if args.jpg == "yes":
    SAVE_IMAGES = True
if args.landmarks == "yes":
    GET_LANDMARKS = True
if args.hog == "yes":
    GET_HOG_FEATURES = True
if args.hog_windows == "yes":
    GET_HOG_WINDOWS_FEATURES = True
if args.onehot == "yes":
    ONE_HOT_ENCODING = True
if args.expressions != "":
    expressions = args.expressions.split(",")
    for i in range(0, len(expressions)):
        label = int(expressions[i])
        if (label >= 0) and (label <= 6):
            SELECTED_LABELS.append(label)
if SELECTED_LABELS == []:
    SELECTED_LABELS = [0, 1, 2, 3, 4, 5, 6]
print(str(len(SELECTED_LABELS)) + " expressions")

original_labels = [0, 1, 2, 3, 4, 5, 6]
new_labels = list(set(original_labels) & set(SELECTED_LABELS))
nb_images_per_label = list(np.zeros(len(new_labels), 'uint8'))
try:
    os.makedirs(OUTPUT_FOLDER_NAME)
except OSError as e:
    if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
        pass
    else:
        raise


def load_fer2013():
    print("importing csv file")
    return pd.read_csv('fer2013.csv')


def get_landmarks(image, rects):
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


def get_new_label(label, one_hot_encoding=False):
    if one_hot_encoding:
        new_label = new_labels.index(label)
        label = list(np.zeros(len(new_labels), 'uint8'))
        label[new_label] = 1
        return label
    else:
        return new_labels.index(label)


def sliding_hog_windows(image):
    hog_windows = []
    for y in range(0, image_height, window_step):
        for x in range(0, image_width, window_step):
            window = image[y:y + window_size, x:x + window_size]
            hog_windows.extend(
                hog(window, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=False))
    return hog_windows


def vis_hogfeature(image):
    hog_features = []
    window_imgs = []
    for y in range(0, image_height, window_size):
        for x in range(0, image_width, window_size):
            window = image[y:y + window_size, x:x + window_size]
            feature, window_hog_img = hog(window,
                                          orientations=8,
                                          pixels_per_cell=(8, 8),
                                          cells_per_block=(1, 1),
                                          visualize=True)
            hog_features.append(feature)
            window_imgs.append(window_hog_img)
    feature, global_img = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    return hog_features, window_imgs, global_img


def visualize_hog():
    VIS_NUM_FOR_EACH_EXPR = 3
    VIS_HOG_FOLDER = os.path.join(VIS_FOLDER_NAME, 'HOG')
    for label in new_labels:
        target = data[data['emotion'] == label]['pixels'].values
        idx = np.random.choice(len(target), VIS_NUM_FOR_EACH_EXPR)
        imgs = [np.fromstring(item, dtype=int, sep=' ').reshape(48, 48).astype('float32') for item in target[idx]]
        imgs = [cv2.resize(item, (image_width, image_height)) for item in imgs]
        hog_results = [vis_hogfeature(item) for item in imgs]
        hog_features = [item[0] for item in hog_results]
        window_hog_imgs = [item[1] for item in hog_results]
        global_hog_imgs = [item[2] for item in hog_results]
        label_folder = os.path.join(VIS_HOG_FOLDER, str(label) + "-" + expr_map[label])
        for i in range(VIS_NUM_FOR_EACH_EXPR):
            target_folder = os.path.join(label_folder, f"{idx[i]}")
            os.makedirs(target_folder, exist_ok=True)
            window_imgs_folder = os.path.join(target_folder, "window_hogs")
            os.makedirs(window_imgs_folder, exist_ok=True)

            ax = plt.subplot(1, 2, 1)
            ax.clear()
            plt.imshow(imgs[i], cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(global_hog_imgs[i])
            plt.savefig(os.path.join(target_folder, f"img_with_hog.png"))
            for step, window_hog_img in enumerate(window_hog_imgs[i]):
                window_img_path = os.path.join(window_imgs_folder, f"window{step}.png")
                ax = plt.subplot(1, 2, 1)
                ax.clear()
                plt.imshow(imgs[i], cmap='gray')
                ax.add_patch(
                    patches.Rectangle((window_size * (step // 3), window_size * (step % 3)),
                                      window_size,
                                      window_size,
                                      fc='none',
                                      ec='g'))
                plt.subplot(1, 2, 2)
                plt.imshow(window_hog_img)
                # plt.show()
                plt.savefig(window_img_path)


def vis_landmarks():
    VIS_NUM_FOR_EACH_EXPR = 3
    LANDMARK_FOLDER = os.path.join(VIS_FOLDER_NAME, 'LANDMARK')
    # loading Dlib predictor and preparing arrays:
    print("preparing predictor")
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    for label in new_labels:
        label_folder = os.path.join(LANDMARK_FOLDER, str(label) + "-" + expr_map[label])
        os.makedirs(label_folder, exist_ok=True)
        target = data[data['emotion'] == label]['pixels'].values
        idx = np.random.choice(len(target), VIS_NUM_FOR_EACH_EXPR)
        imgs = [np.fromstring(item, dtype=int, sep=' ').reshape(48, 48).astype('float32') for item in target[idx]]
        imgs = [cv2.resize(item, (image_width, image_height)) for item in imgs]
        imgs = [item.astype(np.uint8) for item in imgs]
        rectangle = [dlib.rectangle(left=1, top=1, right=299, bottom=299)]
        for step, img in enumerate(imgs):
            target_path = os.path.join(label_folder, f"{idx[step]}.png")
            landmarks = predictor(img, rectangle[0]).parts()
            ax = plt.subplot(1, 1, 1)
            ax.clear()
            plt.imshow(img, cmap='gray')
            for p in landmarks:
                ax.add_patch(patches.Circle((p.x, p.y), 3, fc='g', ec='none'))
            #plt.show()
            plt.savefig(target_path)


def vis_dimension():
    VIS_DIMENSION_FOLDER = os.path.join(VIS_FOLDER_NAME, 'DIMENSION')
    os.makedirs(VIS_DIMENSION_FOLDER, exist_ok=True)
    train_hog = load_data(arg_features='hog')
    train_land = load_data(arg_features='landmarks')
    train_hl = load_data(arg_features='landmarks_and_hog')
    option = ['hog', 'landmarks', 'landmarks_and_hog']
    dimensions = {
        option[2]: train_hl['X'].shape[1],
        option[0]: train_hog['X'].shape[1],
        option[1]: train_land['X'].shape[1],
    }
    classes = dimensions.keys()
    values = dimensions.values()
    plt.subplot(1, 1, 1)
    plt.bar(classes, values)
    plt.xlabel('feature classes')
    plt.ylabel('length of feature')
    plt.title('Feature Length')
    TARGET_NAME = os.path.join(VIS_DIMENSION_FOLDER, 'feature_length_cmp.png')
    plt.savefig(TARGET_NAME)


# data = load_fer2013()
vis_dimension()