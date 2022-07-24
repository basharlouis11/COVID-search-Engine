import math
import cv2
import os
import shutil
import subprocess
import numpy as np
import glob
import ntpath


#the images should have the same size tio be feeded in the convolutional neural network
def resize_image(image):
    dimensions = (224, 224)
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)


#I applied the adaptive Histogram Equalization to improve the contrast of the images in the dataset
def enhance_image_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = clahe.apply(gray)
    rgb_enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)
    return rgb_enhanced_image


#combined function between both previous functions 
def batch_enhancement_and_resize(source_path, target_path, extension="png", process_all=False, batch_size=100):
    for idx, entry in enumerate(os.listdir(source_path)):
        if idx < batch_size or process_all:
            img = cv2.imread(os.path.join(source_path, "./", entry))
            img = enhance_image_contrast(resize_image(img))
            img = resize_image(img)
            cv2.imwrite(os.path.join(target_path, "./", "{}.{}".format(idx, extension)), img)
        else:
            break

            
# We use lungmask trained with U-net neural network to segment the lungs and generate masks https://github.com/JoHof/lungmask


def generate_masks(source_path, target_path):
    for entry in os.listdir(source_path):
        model = "R231CovidWeb"
        process = subprocess.Popen(["lungmask",
                                    os.path.join(source_path, "./", entry),
                                    os.path.join(target_path, "./", entry),
                                    "--modelname",
                                    model,
                                    "--noHU"])
        process.wait()


def apply_mask(image, mask):
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, black_and_white_mask = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(image, image, mask=black_and_white_mask)


def batch_apply_mask(image_source_path, mask_source_path, target_path):
    for entry in os.listdir(image_source_path):
        image = cv2.imread(os.path.join(image_source_path, "./", entry))
        mask = cv2.imread(os.path.join(mask_source_path, "./", entry))
        masked_image = apply_mask(image, mask)
        cv2.imwrite(os.path.join(target_path, "./", entry), masked_image)


def create_or_refresh_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def generate_training_and_test_sets(training_path="../data/training",
                                    test_path="../data/test", masked_covid_path="../data/masked/COVID",
                                    masked_non_covid_path="../data/masked/non-COVID"):
    create_or_refresh_directory(training_path)
    create_or_refresh_directory(os.path.join(training_path, './', 'COVID'))
    create_or_refresh_directory(os.path.join(training_path, './', 'non-COVID'))
    create_or_refresh_directory(test_path)
    create_or_refresh_directory(os.path.join(test_path, './', 'COVID'))
    create_or_refresh_directory(os.path.join(test_path, './', 'non-COVID'))

    training_set_percentage = 0.8
    covid_training_set_size = math.floor(training_set_percentage * len(os.listdir(masked_covid_path)))
    non_covid_training_set_size = math.floor(training_set_percentage * len(os.listdir(masked_non_covid_path)))

    for idx, entry in enumerate(os.listdir(masked_covid_path)):
        if idx < covid_training_set_size:
            shutil.copy(os.path.join(masked_covid_path, "./", entry),
                        os.path.join(training_path, "./COVID", "covid_" + entry))
        else:
            shutil.copy(os.path.join(masked_covid_path, "./", entry), os.path.join(test_path, "./COVID", "covid_" + entry))

    for idx, entry in enumerate(os.listdir(masked_non_covid_path)):
        if idx < non_covid_training_set_size:
            shutil.copy(os.path.join(masked_non_covid_path, "./", entry),
                        os.path.join(training_path, "./non-COVID", "non_covid_" + entry))
        else:
            shutil.copy(os.path.join(masked_non_covid_path, "./", entry),
                        os.path.join(test_path, "./non-COVID", "non_covid_" + entry))


def read_images(path, extension):
    image_array = []
    image_labels = []
    image_paths = []
    for entry in glob.glob(path+"/**/*.{}".format(extension), recursive=True):
        # image_path = os.path.join(path, entry)
        image_paths.append(entry)
        # 1 for covid, 0 for non covid
        label = 1
        if "non" in ntpath.basename(entry):
            label = 0
        image_labels.append(label)
        image = cv2.imread(entry, cv2.IMREAD_COLOR)
        normalized_image = np.zeros((224, 224))
        normalized_image = cv2.normalize(image, normalized_image, 0, 1, cv2.NORM_MINMAX)
        image_array.append(normalized_image)
    return np.array(image_array), image_labels, image_paths


def process_images():
    raw_covid_directory = "../data/raw/COVID"
    raw_non_covid_directory = "../data/raw/non-COVID"
    enhanced_covid_directory = "../data/enhanced/COVID"
    enhanced_non_covid_directory = "../data/enhanced/non-COVID"
    mask_covid_directory = "../data/mask/COVID"
    mask_non_covid_directory = "../data/mask/non-COVID"
    masked_covid_directory = "../data/masked/COVID"
    masked_non_covid_directory = "../data/masked/non-COVID"

    # Enhance and resize images
    print("Enhancing and resizing images")
    create_or_refresh_directory("../data/enhanced")
    os.mkdir(enhanced_covid_directory)
    os.mkdir(enhanced_non_covid_directory)
    batch_enhancement_and_resize(raw_covid_directory, enhanced_covid_directory, process_all=True)
    batch_enhancement_and_resize(raw_non_covid_directory, enhanced_non_covid_directory, process_all=True)

    # Generate masks
    print("Generating masks")
    create_or_refresh_directory("../data/mask")
    os.mkdir(mask_covid_directory)
    os.mkdir(mask_non_covid_directory)
    generate_masks(enhanced_covid_directory, mask_covid_directory)
    generate_masks(enhanced_non_covid_directory, mask_non_covid_directory)

    # Apply masks
    print("Applying masks")
    create_or_refresh_directory("../data/masked")
    os.mkdir(masked_covid_directory)
    os.mkdir(masked_non_covid_directory)
    batch_apply_mask(enhanced_covid_directory, mask_covid_directory, masked_covid_directory)
    batch_apply_mask(enhanced_non_covid_directory, mask_non_covid_directory, masked_non_covid_directory)

    # Training and test set
    print("Generating training and test sets")
    generate_training_and_test_sets(masked_covid_path="../data/enhanced/COVID",
                                    masked_non_covid_path="../data/enhanced/non-COVID")
