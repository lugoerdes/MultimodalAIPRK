import os
import gc
import itertools
import json
import fasttext
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define filepath
sentiment_filepath = 'Sentiment140/training.1600000.processed.noemoticon.csv'
sentiment_train_data = './Sentiment140/train.txt'
sentiment_dev_data = './Sentiment140/dev.txt'
sentiment_test_data = './Sentiment140/test.txt'
confusion_matrix_dir = "ConfusionMatrix/"
solution_json_file_best_models_json = "SolutionJSONFile/best_models.json"
models_dir = "Models/"
# Define parameter
epochs = [15,20]
learning_rates = [0.01,0.5]
word_ngrams = [1]
best_models = []


def convert_csv_to_fast_text_format(sentiment_path, sen_training_data, sen_dev_data, sen_test_data):
    """
    This function converts a csv to the fast text format for the columns label and tweet and then splits and saves it
    into train, dev and test data.
    :param sen_test_data: path to save the test data of the sentiment file
    :param sen_dev_data: path to save the dev data of the sentiment file
    :param sen_training_data: path to save the training data of the sentiment file
    :param sentiment_path: path to the sentiment file
    """
    # load Sentiment140
    df = pd.read_csv(sentiment_path, encoding='latin-1', header=None)
    # name columns
    df.columns = ['polarity', 'id', 'date', 'query', 'user', 'tweet']
    # format Labels to fastText-Format
    df['label'] = df['polarity'].replace({0: '__label__negative', 2: '__label__neutral', 4: '__label__positive'})
    # keep necessary columns
    df_fasttext = df[['label', 'tweet']]
    # split into training, development and test data (70%, 15%, 15%), used random_state to make results reproductable
    train_data, temp_data = train_test_split(df_fasttext, test_size=0.30, random_state=42)
    dev_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    # save data in fastText-Format
    train_data.to_csv(sen_training_data, sep=' ', index=False, header=False)
    dev_data.to_csv(sen_dev_data, sep=' ', index=False, header=False)
    test_data.to_csv(sen_test_data, sep=' ', index=False, header=False)


def train_model(model_dir, sen_dev_data, epoch, lr, word_n_gram):
    """
    This function trains the model with the training data and then test it with the dev data. It then
    :param model_dir: directory of the model
    :param sen_dev_data: path to the sentiment dev data
    :param epoch: Number of complete passes the model makes over training dataset
    :param lr: learnging rate -> controls the size of steps model takes when adjusting weights
    :param word_n_gram: continuous sequences of words given within window size n
    :return: trained model, model file path and calculated f1_score
    """
    # train model
    model = fasttext.train_supervised(input=sentiment_train_data, epoch=epoch, lr=lr, wordNgrams=word_n_gram)
    # test model and calculate f1_score
    f_one_score = calculate_f_one_from_model_after_test(model, sen_dev_data)
    # save model
    model_path = os.path.join(model_dir, f"model_lr{lr}_epoch{epoch}_ngram{word_n_gram}")
    model.save_model(model_path)
    return model, model_path, f_one_score


def calculate_f_one_from_model_after_test(model, sentiment_test_or_dev_data):
    """
    This function calculates the f1_score of a model
    :param sentiment_test_or_dev_data: path to test or dev data
    :param model: model to calculate f1_score
    :return: calculated f1_score, precision and recall of the model
    """
    # test model on test
    result = model.test(sentiment_test_or_dev_data)
    # calculate f1_score
    precision, recall = result[1], result[2]
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def train_and_calculate_f1_score_of_model(model_dir, epoch, lr, word_n_gram, sen_dev_data, sen_test_data):
    """
    This function trains and calculates the f1_scores of a model
    :param sen_test_data: path to the test data
    :param sen_dev_data: path to the dev data
    :param model_dir: directory of the model
    :param epoch: Number of complete passes the model makes over training dataset
    :param lr: learning rate -> controls the size of steps model takes when adjusting weights
    :param word_n_gram: continuous sequences of words given within window size n
    :return: f1_score_on_test/dev, precision, recall, model_path of the model
    """
    model, model_path, f1_score_on_dev = train_model(model_dir, sen_dev_data, epoch, lr, word_n_gram)
    f1_score = calculate_f_one_from_model_after_test(model, sen_test_data)
    return f1_score, model_path, f1_score_on_dev


def create_confusion_matrix(solution_json_file, target_directory):
    """
    This function creates the confusion_matrix of every model that was saved to the solution json file
    :param solution_json_file: used to retrieve model path of the best models
    :param target_directory: directory to save the confusion matrix png
    """
    with open(solution_json_file, "r") as json_file:
        # clear directory and load json_file with three best models
        clear_directory(target_directory)
        data = json.load(json_file)
        # loop through every object in json file
        for obj in data:
            # load current model
            model = fasttext.load_model(obj["model_path"])
            # test model on test data and save the prediction and label
            test_data = open(sentiment_test_data, "r").readlines()
            true_labels = []
            predictions = []
            for line in test_data:
                label, tweet = line.split(" ", 1)
                true_labels.append(label)
                predictions.append(model.predict(tweet.strip())[0][0])
            # create, display and save confusion matrix with labels and prediction
            cm = confusion_matrix(true_labels, predictions)
            labels = ['negative', 'positive']
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot()
            print()
            plt.savefig(
                f"{target_directory}confusion_matrix_model_lr{obj['learning_rate']}_epoch{obj['epoch']}_ngram{obj['ngram']}.png")
            # use garbage collector to prevent computer from crashing
            del model
            gc.collect()


def automatic_parameter_optimization(model_path):
    """
    This function is meant to optimize parameter automatically
    :param model_path: path of the model that should be used
    """
    # train model
    model = fasttext.train_supervised(input=sentiment_train_data, autotuneValidationFile=sentiment_dev_data)
    # calculate f1_score, precision and recall
    f1_score, precision, recall = calculate_f_one_from_model_after_test(model, sentiment_dev_data)
    # save model
    model.save_model(model_path)


def create_three_best_models(model_dir, sentiment_path, training_data, dev_data, test_data,
                             solution_json_directory_file):
    """
    This function trains and saves the three best models found for the parameters
    :param model_dir: directory where the models should be saved
    :param sentiment_path: path to the sentiment csv
    :param training_data: path to the training data
    :param dev_data: path to the dev data
    :param test_data: path to the test data
    :param solution_json_directory_file: json to store solution parameter
    """
    # if one of the data files does not exist, convert the csv to fasttext format and save the files
    if not (os.path.exists(training_data) or os.path.exists(dev_data) or os.path.exists(test_data)):
        convert_csv_to_fast_text_format(sentiment_path, training_data, dev_data, test_data)
    # clear directory with the models
    clear_directory(models_dir)
    # for every lr, epoch and ngram defined, iterate through them to get the best f1_scores and save the models to best_models
    for lr, epoch, ngram in itertools.product(learning_rates, epochs, word_ngrams):
        f1_score_on_test, model_path_on_test, f1_score_on_dev = train_and_calculate_f1_score_of_model(model_dir, epoch,
                                                                                                      lr, ngram,
                                                                                                      dev_data,
                                                                                                      test_data)
        best_models.append((model_path_on_test, f1_score_on_dev, f1_score_on_test, lr, epoch, ngram))
    # sort best_models
    best_models.sort(reverse=True, key=lambda x: x[0])
    # delete every model that is below 3 to have the three best models left and save them in json file
    for i in range(3, len(best_models)):
        model_path_to_delete = best_models[i][0]
        if os.path.exists(model_path_to_delete):
            os.remove(model_path_to_delete)
            print(f"{model_path_to_delete} DELETED.")
        else:
            print(f"{model_path_to_delete} NOT FOUND.")
        del best_models[i]
    save_best_models_in_json(solution_json_directory_file)


def clear_directory(directory):
    """
    This function clears a directory.
    :param directory: path to a directory that should be cleared.
    """
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"DELETION FAILED FROM {file_path}: {e}")
    else:
        os.makedirs(directory)


def save_best_models_in_json(solution_json_directory_file):
    """
    This function saves the best models in a json file
    :param solution_json_directory_file: filepath to save solution in json format
    """
    model_data_list = []
    for row in best_models:
        model_data = {"model_path": row[0],
                      "f1_score_on_dev": row[1],
                      "f1_score_on_test": row[2],
                      "learning_rate": row[3],
                      "epoch": row[4],
                      "ngram": row[5]
                      }
        model_data_list.append(model_data)
    with open(solution_json_directory_file, 'w') as json_file:
        json.dump(model_data_list, json_file, indent=4)


def create_three_best_models_and_confusion_matrix(model_dir, sentiment_path,
                                                  training_data, dev_data, test_data, solution_json_file,
                                                  confusion_matrix_target_dir):
    """
     This function creates the three best models and the confusion matrix for them
    :param model_dir: directory path where to save the model
    :param sentiment_path: path to the csv of the sentiment data
    :param training_data: path to the training data
    :param dev_data: path to the dev data
    :param test_data: path to the test data
    :param solution_json_file: file to save solution in json format
    :param confusion_matrix_target_dir: directory to save confusion matrix png
    :return:
    """
    create_three_best_models(model_dir, sentiment_path, training_data, dev_data, test_data,
                             solution_json_file)
    create_confusion_matrix(solution_json_file, confusion_matrix_target_dir)


create_three_best_models_and_confusion_matrix(models_dir, sentiment_filepath,
                                              sentiment_train_data, sentiment_dev_data, sentiment_test_data,
                                              solution_json_file_best_models_json, confusion_matrix_dir)
