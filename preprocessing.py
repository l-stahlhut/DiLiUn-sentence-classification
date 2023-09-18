import argparse
import os
import pathlib
import re
import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

nlp = spacy.load('de_core_news_md')


"""
How to use the script
e.g. With the emprirically annotated data from iterations 1-3 and the synthetic data from the iteration 1 and 2
without undersampling:
$ python3 preprocessing.py data/raw --emp 1 2 3 --syn 1 2 --undersample

"""

parser = argparse.ArgumentParser(
    prog='main',
    description='Read annotated data, preprocess it, and train the model with it.')
parser.add_argument('datapath', type=pathlib.Path, help='Path of raw training data, e.g. data/raw')
parser.add_argument('-e', '--emp', nargs='+', help='List of Itertions of empirically annotated data to consider '
                                                         'for training, e.g. "--emp 1 2" to consider data that was '
                                                         'annotated in iterations 1 and 2.')
parser.add_argument('-s', '--syn', nargs='+', help='List of Itertions of synthetically generated data to consider '
                                                         'for training, e.g. "--syn 1 2" to consider data that was '
                                                         'annotated in iterations 1 and 2.')
parser.add_argument('-u', '--undersample', action='store_true', help='"True" to undersample data to minority class or '
                                                                     '"False" to work with all data.')
args = parser.parse_args()


def read_data(iterations: list) -> list[dict[str, list[str]]]:
    """Reads data according to the command line arguments.
    iterations: list of iterations (syn/emp) to consider, e.g.  ['syn_I1', 'syn_I2', 'syn_I3', 'emp_I1']

    Returns a dictionary with steps as keys and preprocessed sentences as values."""
    data_paths = [os.path.join('data/raw', step) for step in iterations]
    steps = ["no_step", "step1c", "step1d", "step1e", "step2a", "step2c", "step3a", "step3b", "step3d", "step3e",
             "step3g"]
    steps_dictionary = {"no_step": [],
             "step1c": [],
             "step1d": [],
             "step1e": [],
             "step2a": [],
             "step2c": [],
             "step3a": [],
             "step3b": [],
             "step3d": [],
             "step3e": [],
             "step3g": []}

    for path in data_paths:
        for step in steps:
            filepath = os.path.join(path, step + '.txt')
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    lines = [l.rstrip('\n') for l in lines if l != '\n' and not l[0].isdigit()]  # exclude titles
                    lines = preprocess_text(lines) # apply annas preprocessing
                    steps_dictionary[step].extend(lines)
            except FileNotFoundError:
                # print("File ", filepath, "does not exist.")
                pass

    return steps_dictionary

def preprocess_text(phrases: list[str]) -> list[str]:
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
        tokens_prepro = [tok.lemma_ for tok in tokens_prepro] # todo
        phrase_prepro = ' '.join(tokens_prepro)
        phrase_prepro = re.sub(r'[0-9]', '', phrase_prepro)
        phrases_prepro.append(phrase_prepro)
    return phrases_prepro

def data_to_df(steps_dictionary: list[dict[str, list[str]]]) -> pd.DataFrame:
    """Create a df with class in onee column and string n the other"""
    data = {
        'class': [],
        'sentence': []
    }

    for class_label, sentences in steps_dictionary.items():
        data['class'].extend([class_label] * len(sentences))
        data['sentence'].extend(sentences)

    df = pd.DataFrame(data)

    return df

def train_test_split(df: pd.DataFrame, shortname:str) -> None:
    """Data split of dataframe (80/10/10 for train/test/val)"""
    # create folder
    if not os.path.exists(os.path.join('data/intermediate', shortname)):
        os.makedirs(os.path.join('data/intermediate', shortname))

    #df = pd.read_csv(df) # uncomment in case you're reading a file (not df)
    # get random sample for test
    train = df.sample(frac=0.6, axis=0)
    # get everything but the test sample
    rest = df.drop(index=train.index)
    # split val/test 50/50
    val = rest.sample(frac=0.5, axis=0)
    test = rest.drop(index=val.index)

    # undersample train data
    if args.undersample:
        train = undersample_train_data(train)

    # write to csv
    train.to_csv(os.path.join('data/intermediate', shortname, 'train.csv'), index=False)
    test.to_csv(os.path.join('data/intermediate', shortname, 'test.csv'), index=False)
    val.to_csv(os.path.join('data/intermediate', shortname, 'val.csv'), index=False)

    return train, test, val

def undersample_train_data(train_data: pd.DataFrame) -> pd.DataFrame:
    # Find the class with the minimum number of samples
    min_class_size = min(train_data['class'].value_counts())

    # Create an empty dataframe to store the balanced data
    balanced_data = pd.DataFrame(columns=['class', 'sentence'])

    # Undersample each class to have the same number of samples as the minimum class
    for class_label in train_data['class'].unique():
        class_samples = train_data[train_data['class'] == class_label]
        balanced_samples = resample(class_samples, n_samples=min_class_size, random_state=42)
        balanced_data = pd.concat([balanced_data, balanced_samples])

    train_data = balanced_data

    return train_data

def main():
    # Chose which data to work with
    print('emp'+''.join(args.emp))
    # print('emp' + ''.join(args.emp) + '_syn' + ''.join(args.syn))
    if args.emp is not None:
        print("Working with the following data:\n- empirically annotated data from iterations ", args.emp)
        iterations = ["emp_I"+x for x in args.emp]
        shortname = 'emp'+''.join(args.emp)
        if args.syn is not None:
            print("- synthetic data from iterations ", args.syn)
            paths_synthetic = ["syn_I" + x for x in args.syn]
            iterations = paths_synthetic + iterations
            shortname = shortname + '_syn' + ''.join(args.syn)
            print("-> Working with the following data for a mixed approach: ", iterations)
            print("Shortname for saving preprocessed data: ", shortname)
        else:
            print("-> Working with empirically annotated data only (no mixed approach): ", iterations)
    else:
        print("Please specify which iterations of the empirical data to work with.")

    # read and preprocess data, save as csv in data/intermediate
    print("Preprocessing texts...")
    steps_dictionary = read_data(iterations)
    df = data_to_df(steps_dictionary)
    print("Preprocessing finished.")
    value_counts = df['class'].value_counts().reset_index()
    print("\nNumber of sentences per class:\n", value_counts)
    print("\nSplitting dataset into train, test, val...")
    train, test, val = train_test_split(df, shortname)
    print("Data split complete.\n\nNumber of sentences per class in train data (after optional undersampling)"
          ":\n", train['class'].value_counts().reset_index())


if __name__ == '__main__':
    main()