#CNN with deepset/gbert-large

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
from transformers import AutoTokenizer, TFAutoModel
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import re
import spacy

nlp = spacy.load('de_core_news_md')

print("gpu support", tf.config.list_physical_devices('GPU'))
#
# parser = argparse.ArgumentParser(
#     prog='main',
#     description='Read annotated data, preprocess it, and train the model with it.')
# parser.add_argument('-e', '--emp', nargs='+', help='List of Itertions of empirically annotated data to consider '
#                                                          'for training, e.g. "--emp 1 2" to consider data that was '
#                                                          'annotated in iterations 1 and 2.')
# parser.add_argument('-s', '--syn', nargs='+', help='List of Itertions of synthetically generated data to consider '
#                                                          'for training, e.g. "--syn 1 2" to consider data that was '
#                                                          'annotated in iterations 1 and 2.')
#
# args = parser.parse_args()


class Dataset:
    def __init__(self, data_path):
        # read csv's into dataframes
        self.train = pd.read_csv(os.path.join(data_path, "train.csv"), encoding='utf8',
                                 dtype={'text_id': str, 'sentence': str, 'label': str}, index_col=False).dropna()
        self.val = pd.read_csv(os.path.join(data_path, "val.csv"), encoding='utf8',
                               dtype={'text_id': str, 'sentence': str, 'label': str}, index_col=False).dropna()
        self.test = pd.read_csv(os.path.join(data_path, "test.csv"), encoding='utf8',
                                dtype={'text_id': str, 'sentence': str, 'label': str}, index_col=False).dropna()
        # save list of sentences and series of corresponding labels
        self.sentences_train_raw, self.labels_train = self.get_target_sentence(self.train)
        self.sentences_val_raw, self.labels_val = self.get_target_sentence(self.val)
        self.sentences_test_raw, self.labels_test = self.get_target_sentence(self.test)
        # map labels to integers
        self.labels_train = self.map_labels(self.labels_train)
        self.labels_val = self.map_labels(self.labels_val)
        self.labels_test = self.map_labels(self.labels_test)
        # preprocess sentences
        self.sentences_train = self.preprocess_sents(self.sentences_train_raw)
        self.sentences_val = self.preprocess_sents(self.sentences_val_raw)
        self.sentences_test = self.preprocess_sents(self.sentences_test_raw)
        # encode data
        # specify model and tokenizer
        self.model_name = "deepset/gbert-large"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = TFAutoModel.from_pretrained(self.model_name)
        self.max_length = 50 #TODO
        # Tokenize and pad training sentences
        self.sentences_train_embeddings = self.encode_data(self.sentences_train)
        self.sentences_train_padded = tf.keras.preprocessing.sequence.pad_sequences(self.sentences_train_embeddings,
                                                                                maxlen=self.max_length, padding='post',
                                                                                dtype='float32')
        # Tokenize and pad validation sentences
        self.sentences_val_embeddings = self.encode_data(self.sentences_val)
        self.sentences_val_padded = tf.keras.preprocessing.sequence.pad_sequences(self.sentences_val_embeddings,
                                                                              maxlen=self.max_length, padding='post',
                                                                              dtype='float32')
        # Tokenize and pad test sentences
        self.sentences_test_embeddings = self.encode_data(self.sentences_test)
        self.sentences_test_padded = tf.keras.preprocessing.sequence.pad_sequences(self.sentences_test_embeddings,
                                                                               maxlen=self.max_length, padding='post',
                                                                               dtype='float32')

    def get_target_sentence(self, df):
      """
      input: dataframe that contains sentences and labels.
      output: labels as pandas.core.series.Series & sentences as list
      """
      labels = df.pop("class").to_numpy()
      sentences = df["sentence"].to_list()
      return sentences, labels

    def preprocess_sents(self, phrases):
        """
        Preprocessing function from Anna.
        Performs preprocessing of the phrases. The preprocessing includes:
        - lemmatization
        - removing stopwords
        - removing OOV words
        - removing named entities
        - removing custom stopwords
        - removing numbers
        :param phrases: a list of phrases to be pre-processed
        :return: a list of pre-processed phrases
        """
        phrases_prepro = []
        custom_stopwords = ['bzw', 'vgl', 'evt', 'evtl']
        for ph in phrases:
            ph_wo_punct = re.sub(r'[^\w\s]', '', ph)
            phrase = nlp(ph_wo_punct)
            tokens = [tok for tok in phrase]
            tokens_prepro = [tok for tok in tokens if tok.is_stop is False]
            tokens_prepro = [tok for tok in tokens_prepro if tok.is_oov is False]
            tokens_prepro = [tok for tok in tokens_prepro if tok.text not in [ne.text for ne in phrase.ents]]
            tokens_prepro = [tok for tok in tokens_prepro if tok.text not in custom_stopwords]
            tokens_prepro = [tok.lemma_ for tok in tokens_prepro]  # todo
            phrase_prepro = ' '.join(tokens_prepro)
            phrase_prepro = re.sub(r'[0-9]', '', phrase_prepro)
            phrases_prepro.append(phrase_prepro)

        return phrases_prepro

    def encode_data(self, sentences):
        # Tokenize the sentences using the tokenizer
        tokenized = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_length,
                                   return_tensors="tf")  # TODO alle sätze nehmen, server
        # Get the model embeddings for the tokenized sentences
        embeddings = self.model(tokenized.input_ids, attention_mask=tokenized.attention_mask)[0]
        # Return the embeddings
        return embeddings.numpy()

    def map_labels(self, labels):
        """Map categorical labels to integers."""
        label_to_int = {
            "no_step": 0,
            "step1c": 1,
            "step1d": 2,
            "step1e": 3,
            "step2a": 4,
            "step2c": 5,
            "step3a": 6,
            "step3b": 7,
            "step3d": 8,
            "step3g": 9
        }
        integer_labels = [label_to_int[label] for label in labels]
        return pd.Series(integer_labels)


    def map_labels_old(self, labels):
        """Map labels to integers."""
        label_to_int = {label: idx for idx, label in enumerate(set(labels))}
        integer_labels = [label_to_int[label] for label in labels]
        return pd.Series(integer_labels)


def get_nn_model(
        n_sentences, # sentences_train_padded.shape[1]
        filter_size, # sentences_train_padded.shape[2]
        num_classes, # number of classes (10)
    ):
    # model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_sentences, filter_size)),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax') # multiclass classification
    ])

    model.compile(
        optimizer='adam', loss='sparse_categorical_crossentropy', # todo ausprobieren
        metrics=['accuracy'])

    return model


def train_model(
        model,
        sentences_train, # sentences_train_padded
        labels_train,
        sentences_val, # sentences_val_padded2
        labels_val,
        model_name,
        dataset,
    ):
    history = model.fit(sentences_train, labels_train,
                            epochs=1,
                            batch_size=32,
                            validation_data=(sentences_val, labels_val))

    # check for overfitting: validation loss plot
    print("Check for overfitting...")
    # plot_result(history, dataset)
    plt.plot(history.history['accuracy'], label="Training accuracy")
    plt.plot(history.history['val_accuracy'], label="Validation accuracy")
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    # save plot to file
    if not os.path.exists(os.path.join("nn/reports", dataset)):
        os.makedirs(os.path.join("nn/reports", dataset))
    plt.savefig(os.path.join("nn/reports", dataset, "val_loss.png"))

    # save model to a file
    if not os.path.exists("nn/models"):
        os.makedirs("nn/models")
    model.save(os.path.join("nn/models", model_name + ".h5"))

def plot_result(history, dataset):
    """Plot train and val loss during training."""
    plt.plot(history.history['accuracy'], label ="Training accuracy")
    plt.plot(history.history['val_accuracy'], label ="Validation accuracy")
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    # plt.show()

    # save plot to file
    if not os.path.exists(os.path.join("nn/reports", dataset)):
        os.makedirs(os.path.join("nn/reports", dataset))
    plt.savefig(os.path.join("nn/reports", dataset, "val_loss.png"))

    return None


def evaluation_report(model, X_test, y_true, dataset):
    """Returns classification report for a ten-way classification model.
    model = the model we want to evaluate;
    X_test = the test data;
    y_true = true class labels (not numeric labels).
    dataset = the name of the dataset.
    """
    # # accuracy
    # loss, accuracy = model.evaluate(X_test, y_true)
    # print('Test loss:', loss) # todo delete?
    # print('Test accuracy:', accuracy)
    # print("\n")

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels (0 to 9) todo uncoment

    # Create a list of class names in the same order as the numeric labels
    class_names = ["no_step", "step1c", "step1d", "step1e", "step2a", "step2c", "step3a", "step3b", "step3d", "step3g"]

    # Generate a classification report with class names
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    #report = classification_report(y_true, y_pred, target_names=class_names)

    # print("Evaluation report: ")
    # print(report)

    # Save the report to a file
    if not os.path.exists(os.path.join("nn/reports", dataset)):
        os.makedirs(os.path.join("nn/reports", dataset))
    with open(os.path.join("nn/reports", dataset, "evaluation_report.txt"), 'w') as file:
        file.write(report)

    return None


def plot_confusion_matrix(model, X_test, y_test, dataset):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
    y_true = y_test.to_numpy()

    # Modify the label dictionary to include all ten classes
    label_dict = {
        0: "no_step",
        1: "step1c",
        2: "step1d",
        3: "step1e",
        4: "step2a",
        5: "step2c",
        6: "step3a",
        7: "step3b",
        8: "step3d",
        9: "step3g"
    }

    # Create the confusion matrix using the label names
    cm = confusion_matrix(y_true, y_pred_classes)

    # Set up the heatmap
    sns.set(font_scale=1.0) # todo adjust font size
    fontdict = {'fontsize': 8}
    ax = sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='d', xticklabels=label_dict.values(), yticklabels=label_dict.values(), cbar_kws={'label': 'Count'}, annot_kws={"size": 16})

    # ax = sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_dict.values(), yticklabels=label_dict.values(), cbar_kws={'label': 'Count'}, annot_kws={"size": 16})
    # ax = sns.heatmap(cm, annot=True)


    ax.set_xlabel('Predicted Step', fontdict=fontdict)
    ax.set_ylabel('Actual Step', fontdict=fontdict)
    ax.set_title('Confusion Matrix CNN, Ten-Way Classification')
    ax.tick_params(axis='both', which='major', labelsize=12)
    # plt.show()

    # Save plot to file
    if not os.path.exists(os.path.join("nn/reports", dataset)):
        os.makedirs(os.path.join("nn/reports", dataset))
    # # plt.savefig(os.path.join("nn/reports", dataset, "confusion_matrix.png")) # todo delete
    ax.figure.savefig(os.path.join("nn/reports", dataset, "confusion_matrix.png"))
    # ax.figure.savefig("output.png")


def save_classified_test_sents(model, dataset, test_sents_raw, test_sents_preprocessed, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
    y_true = y_test.to_numpy()

    label_dict = {
        0: "no_step",
        1: "step1c",
        2: "step1d",
        3: "step1e",
        4: "step2a",
        5: "step2c",
        6: "step3a",
        7: "step3b",
        8: "step3d",
        9: "step3g"
    }

    y_true = [label_dict[label] for label in y_true]
    y_pred_classes = [label_dict[label] for label in y_pred_classes]

    # turn relevant columns into d and pd dataframe
    d = {
        'Raw test sentence': test_sents_raw,
        'Preprocessed test sentence': test_sents_preprocessed,
        'True label': y_true,
        'Predicted label': y_pred_classes
    }
    df = pd.DataFrame(d)

    # save as csv
    if not os.path.exists(os.path.join("nn/reports", dataset)):
        os.makedirs(os.path.join("nn/reports", dataset))
    df.to_csv(os.path.join("nn/reports", dataset, "classified_sentences.csv"), sep='\t', encoding='utf-8')

    return df.head()



# def main():
#     # Get path for specified dataset version from command line arguments
#     # empirical data is mandatory, mixed is optional
#     if args.emp is not None:
#         dataset = 'emp' + ''.join(args.emp)
#         if args.syn is not None:
#             dataset = dataset + '_syn' + ''.join(args.syn)
#     else:
#         print("Please specify which data to work with.")
#
#     dataset_path = os.path.join('./data/intermediate', dataset)
#     print("Working with the data saved in: {}".format(dataset_path))
#
#     # get dataset
#     D = Dataset(dataset_path)
#
#     # create model
#     model = get_nn_model(
#         D.sentences_train_padded.shape[1],  # sentences_train_padded.shape[1]
#         D.sentences_train_padded.shape[2],  # sentences_train_padded.shape[2]
#         num_classes=10,
#     )
#
#     # train model
#     train_model(
#         model,
#         D.sentences_train_padded,  # sentences_train_padded
#         D.labels_train,
#         D.sentences_val_padded,  # sentences_val_padded2
#         D.labels_val,
#         model_name="model_" + dataset,  # e.g. model_emp123 -> shows which dataset it was trained with
#     )
#
#     # evaluate model
#     evaluation_report(model, D.sentences_test_padded, D.labels_test, dataset)
#     # confusion matrix
#     plot_confusion_matrix(model, D.sentences_test_padded, D.labels_test, dataset)
#
#
#
# if __name__ == "__main__":
#     main()



