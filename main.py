from src.feature_extraction import extract_features_vgg16, calculate_and_plot_precision_and_recall, rank_documents, \
    extract_features_vgg16_fine_tuned
from src.image_processing import read_images, process_images
from joblib import load
from feature_extraction import evaluate_features, fine_tune_vgg16_model, train_vgg16_model, refine_features


def evaluate_retrieval():
    training_set_images, training_set_labels, training_set_image_paths = read_images("../data/training", "png")
    test_set_images, test_set_labels, test_set_image_paths = read_images("../data/test", "png")
    training_set_features = extract_features_vgg16(training_set_images, "./training_features.joblib",
                                                   save_features=True)
    test_set_features = extract_features_vgg16(test_set_images, "./test_features.joblib",
                                               save_features=True)
    calculate_and_plot_precision_and_recall(training_set_features,
                                            training_set_labels,
                                            training_set_image_paths,
                                            test_set_features,
                                            test_set_labels,
                                            top_k_step=10,
                                            top_k_percentage=0.3
                                            )


def evaluate_feature_selection():
    training_set_images, training_set_labels, training_set_image_paths = read_images("../data/training", "png")
    training_set_features = extract_features_vgg16(training_set_images, "./training_features.joblib",
                                                   save_features=True)
    evaluate_features(training_set_features, training_set_labels)


def evaluate_retrieval_with_feature_selection():

    training_set_images, training_set_labels, training_set_image_paths = read_images("../data/training", "png")
    test_set_images, test_set_labels, test_set_image_paths = read_images("../data/test", "png")
    training_set_features = extract_features_vgg16(training_set_images, "./training_features.joblib",
                                                   save_features=True)
    test_set_features = extract_features_vgg16(test_set_images, "./test_features.joblib",
                                               save_features=True)
    training_set_features, test_set_features = refine_features(training_set_features,
                                                               training_set_labels,
                                                               test_set_features, 18.5)
    calculate_and_plot_precision_and_recall(training_set_features,
                                            training_set_labels,
                                            training_set_image_paths,
                                            test_set_features,
                                            test_set_labels,
                                            top_k_step=10,
                                            top_k_percentage=0.3
                                            )


def evaluate_retrieval_with_feature_selection_and_finetuning():
    training_set_images, training_set_labels, training_set_image_paths = read_images("../data/training", "png")
    test_set_images, test_set_labels, test_set_image_paths = read_images("../data/test", "png")
    vgg_model = fine_tune_vgg16_model(3,  0.00001)
    train_vgg16_model("../data/training", "../data/test", vgg_model, 64)
    training_set_features = extract_features_vgg16_fine_tuned(training_set_images, vgg_model)
    test_set_features = extract_features_vgg16_fine_tuned(test_set_images, vgg_model)
    training_set_features, test_set_features = refine_features(training_set_features,
                                                               training_set_labels,
                                                               test_set_features, 18.5)
    calculate_and_plot_precision_and_recall(training_set_features,
                                            training_set_labels,
                                            training_set_image_paths,
                                            test_set_features,
                                            test_set_labels,
                                            top_k_step=10,
                                            top_k_percentage=0.3
                                            )


def main():
    is_running = True
    option = None
    while is_running:
        try:
            print("Make sure that your raw data is at ./data/raw\n"
                  "Select the operation\n"
                  "\n"
                  "1.Image processing \n"
                  "2.Evaluate retrieval \n"
                  "3.Evaluate feature selection\n"
                  "5.Evaluate retrieval with feature selection\n"
                  "5.Evaluate retrieval with feature selection and fine tuning\n"
                  "6.Exit"
                  )
            option = int(input('Please select an operation: '))
        except:
            print("Invalid option")
            continue
        if 1 <= option <= 6:
            if option == 1:
                process_images()
            elif option == 2:
                evaluate_retrieval()
            elif option == 3:
                evaluate_feature_selection()
            elif option == 4:
                evaluate_retrieval_with_feature_selection()
            elif option == 5:
                evaluate_retrieval_with_feature_selection_and_finetuning()
            else:
                exit(0)
        else:
            print("Invalid option")
            continue


if __name__ == "__main__":
    main()
