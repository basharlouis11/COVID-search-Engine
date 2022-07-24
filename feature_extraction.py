import math
import os
import keras.optimizers
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import svm
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from livelossplot.inputs.keras import PlotLossesCallback


def extract_features_vgg16(image_array, features_path, save_features=False):
    # Omit the top dense layer since we are only using it to extract the features
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    feature_extractor = vgg_model.predict(image_array)
    features = feature_extractor.reshape(feature_extractor.shape[0], -1)
    if save_features:
        if os.path.exists(features_path):
            os.remove(features_path)
        dump(features, features_path)
    return features


def extract_features_vgg19(image_array, features_path, save_features=False):
    # Omit the top dense layer since we are only using it to extract the features
    vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    feature_extractor = vgg_model.predict(image_array)
    features = feature_extractor.reshape(feature_extractor.shape[0], -1)
    if save_features:
        if os.path.exists(features_path):
            os.remove(features_path)
        dump(features, features_path)
    return features


def extract_features_vgg16_fine_tuned(image_array, vgg_model):
    vgg_model.load_weights('vgg16_model_fine_tuned.weights.best.hdf5')
    topless_vgg_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block5_pool').output)
    feature_extractor = topless_vgg_model.predict(image_array)
    features = feature_extractor.reshape(feature_extractor.shape[0], -1)
    return features


def evaluate_features(training_set_features, training_set_labels):
    clf = Pipeline(
        [
            ("anova", SelectPercentile(chi2)),
            ("scaler", StandardScaler()),
            ("svc", SVC(gamma="auto"))
        ])
    score_means = list()
    score_stds = list()
    percentiles = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    for percentile in percentiles:
        clf.set_params(anova__percentile=percentile)
        this_scores = cross_val_score(clf, training_set_features, training_set_labels)
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())

    plt.errorbar(percentiles, score_means, np.array(score_stds))
    plt.title("Performance of the SVM-Anova varying the percentile of features selected")
    plt.xticks(np.linspace(0, 100, 12, endpoint=True))
    plt.xlabel("Percentile")
    plt.ylabel("Accuracy Score")
    plt.axis("tight")
    plt.show()


def calculate_and_plot_precision_and_recall(training_set_features,
                                            training_set_labels,
                                            training_set_image_paths,
                                            test_set_features,
                                            test_set_labels,
                                            top_k_step=10,
                                            top_k_percentage=0.3
                                            ):
    # The maximum value for top k elements would be half of the training set
    top_k = []
    for k in range(math.floor((training_set_features.shape[0] * top_k_percentage) / top_k_step)):
        top_k.append((k + 1) * top_k_step)

    results = np.ndarray(shape=(test_set_features.shape[0], len(top_k)), dtype=object)

    for document_id in range(test_set_features.shape[0]):
        ranked_results = rank_documents(test_set_features[document_id],
                                        training_set_features,
                                        training_set_labels,
                                        training_set_image_paths,
                                        refine_with_classifier=False,
                                        )
        ground_truth = test_set_labels[document_id]
        total_number_of_relevant_documents = 0
        for i in training_set_labels:
            if i == ground_truth:
                total_number_of_relevant_documents += 1
        for k in top_k:
            index = int((k / top_k_step) - 1)
            tp, fp = 0, 0
            for entry in ranked_results[0:k]:
                if ground_truth == entry[0]:
                    tp += 1
                else:
                    fp += 1
            precision_at_k = tp / (tp + fp)
            recall_at_k = tp / total_number_of_relevant_documents
            results[document_id][index] = (precision_at_k, recall_at_k)
    average_precision = []
    average_recall = []
    print()
    for entry in range(results.shape[1]):
        precision_sum = 0
        recall_sum = 0
        for tup in results[:, entry]:
            precision_sum = precision_sum + tup[0]
            recall_sum = recall_sum + tup[1]
        average_precision.append(precision_sum / results.shape[0])
        average_recall.append(recall_sum / results.shape[0])

    print(average_precision)
    plt.figure(figsize=(6, 6), layout='constrained')
    plt.plot(top_k, average_precision, label='Precision@k')
    plt.plot(top_k, average_recall, label='Recall@k')
    plt.xticks(np.linspace(0, top_k[-1], top_k_step))
    plt.yticks(np.linspace(0, 1, 25))
    plt.xlabel('Top k')
    plt.ylabel('Precision and Recall')
    plt.title("Performance at retrieving {}% of ranked documents".format(int(top_k_percentage * 100)))
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 6), layout='constrained')
    plt.plot(average_recall, average_precision, label='Precision')
    plt.yticks(np.linspace(0, 1, 25))
    plt.xlabel('recall')
    plt.ylabel('Precision')
    plt.title("Performance at retrieving {}% of ranked documents".format(int(top_k_percentage * 100)))
    plt.legend()
    plt.show()


def refine_features(training_set_features, training_set_labels, test_set_features, percentile):
    feature_indexes = []
    for i in range(training_set_features.shape[1]):
        feature_indexes.append(i)
    feature_selection = SelectPercentile(chi2, percentile=percentile)
    feature_selection.fit_transform(training_set_features, training_set_labels)
    selected_features = np.array(feature_selection.get_feature_names_out(feature_indexes)).astype(int)
    return training_set_features[:, selected_features], test_set_features[:, selected_features]


def rank_documents(query,
                   documents,
                   labels,
                   image_paths,
                   refine_with_classifier=False,
                   predicted_query_label=None):
    ranked_documents = []
    for document_id in range(documents.shape[0]):
        if refine_with_classifier:
            if predicted_query_label != labels[document_id]:
                continue
        ranked_documents.append((labels[document_id],
                                 cosine_similarity(query.reshape(1, -1), documents[document_id].reshape(1, -1)),
                                 image_paths[document_id]
                                 ))
    ranked_documents.sort(key=lambda tup: tup[1], reverse=True)
    return ranked_documents


def train_svm_classifier(training_set_features,
                         training_set_labels,
                         svm_classifier_path="./svm_classifier.joblib",
                         save_classifier=False):
    classifier = svm.SVC()
    classifier.fit(training_set_features, training_set_labels)
    # We'll save classifier after training
    if save_classifier:
        if os.path.exists(svm_classifier_path):
            os.remove(svm_classifier_path)
        dump(classifier, svm_classifier_path)
    return classifier


def fine_tune_vgg16_model(fine_tuning_layers, learning_rate):
    conv_base = VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=(224, 224, 3))
    if fine_tuning_layers > 0:
        for layer in conv_base.layers[:-fine_tuning_layers]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(4096, activation='relu')(top_model)
    top_model = Dense(1072, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(1, activation='sigmoid')(top_model)
    model = Model(inputs=conv_base.input, outputs=output_layer)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_vgg16_model(train_data_dir, test_data_dir, vgg_model, batch_size):
    # Using image augmentation methods from Keras
    train_generator = ImageDataGenerator(
                                         rescale=1./255,
                                         # horizontal_flip=True,
                                         # vertical_flip=True,
                                         validation_split=0.15,
                                         preprocessing_function=preprocess_input)  # VGG16 preprocessing
    #train_generator = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)  # VGG16 preprocessing
    test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    traingen = train_generator.flow_from_directory(train_data_dir,
                                                   target_size=(224, 224),
                                                   class_mode='binary',
                                                   color_mode='rgb',
                                                   subset='training',
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   seed=42)
    validgen = train_generator.flow_from_directory(train_data_dir,
                                                   target_size=(224, 224),
                                                   class_mode='binary',
                                                   color_mode='rgb',
                                                   subset='validation',
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   seed=42)

    testgen = test_generator.flow_from_directory(test_data_dir,
                                                 target_size=(224, 224),
                                                 class_mode='binary',
                                                 batch_size=1,
                                                 color_mode='rgb',
                                                 shuffle=False,
                                                 seed=42)

    n_steps = traingen.samples // batch_size
    n_val_steps = validgen.samples // batch_size
    n_epochs = 50
    plot_loss_1 = PlotLossesCallback()

    # ModelCheckpoint callback - save best weights
    tl_checkpoint_1 = ModelCheckpoint(filepath='vgg16_model_fine_tuned.weights.best.hdf5',
                                      save_best_only=True,
                                      mode='min',
                                      monitor='val_loss',
                                      verbose=1)

    # EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss',
                               patience=10,
                               restore_best_weights=True,
                               mode='min')
    vgg_history = vgg_model.fit(traingen,
                                batch_size=batch_size,
                                epochs=n_epochs,
                                validation_data=validgen,
                                steps_per_epoch=n_steps,
                                validation_steps=n_val_steps,
                                callbacks=[tl_checkpoint_1, early_stop, plot_loss_1],
                                verbose=1)
    # vgg_model.load_weights('vgg16_model_fine_tuned.weights.best.hdf5')  # initialize the best trained weights
    # true_classes = testgen.classes
    # vgg_preds_ft = vgg_model.predict(testgen)
    # vgg_pred_classes_ft = np.argmax(vgg_preds_ft, axis=1)
    # vgg_acc_ft = accuracy_score(true_classes, vgg_pred_classes_ft)
    # print("VGG16 Model Accuracy with Fine-Tuning: {:.2f}%".format(vgg_acc_ft * 100))
