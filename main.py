import argparse
import os
import sys
import pathlib
from preprocessing import read_data, preprocess_text, data_to_df, train_test_split, undersample_train_data
from nn.train_model import Dataset, get_nn_model, train_model, plot_result, evaluation_report, plot_confusion_matrix, save_classified_test_sents


"""
How to use the script
e.g. With the emprirically annotated data from iterations 1-3 and the synthetic data from the iteration 1 and 2
without undersampling:
$ python3 main.py data/raw --emp 1 2 3 --syn 1 2 --undersample

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



def main():
    # preprocessing .............................................
    # Chose which data to work with
    print('emp'+''.join(args.emp))
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

    # write log info to txt file
    # Define the file path for the output file
    output_file_path = os.path.join("notes", shortname, "log_file_" + shortname + ".txt")
    # Create the 'notes' directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Redirect the standard output to the text file
    with open(output_file_path, "w") as output_file:
        sys.stdout = output_file

        # read and preprocess data, save as csv in data/intermediate
        print("Preprocessing texts...")
        steps_dictionary = read_data(iterations)
        df = data_to_df(steps_dictionary)
        print("Preprocessing finished.")
        value_counts = df['class'].value_counts().reset_index()
        print("\nNumber of sentences per class:\n", value_counts)
        print("\nSplitting dataset into train, test, val...")
        train, test, val = train_test_split(df, shortname, args.undersample)
        print("Data split complete.\n\nNumber of sentences per class in train data (after optional undersampling)"
              ":\n", train['class'].value_counts().reset_index())

        # train model
        # Get path for specified dataset version from command line arguments
        # empirical data is mandatory, mixed is optional
        if args.emp is not None:
            dataset = 'emp' + ''.join(args.emp)
            if args.syn is not None:
                dataset = dataset + '_syn' + ''.join(args.syn)
        else:
            print("Please specify which data to work with.")

        dataset_path = os.path.join('data/intermediate_old', dataset)
        print("Working with the data saved in: {}".format(dataset_path))

    # Reset the standard output back to the terminal
    sys.stdout = sys.__stdout__

    # get dataset
    D = Dataset(dataset_path)
    # print(D.sentences_train_raw[0:2])
    # print(D.sentences_train[0:2])
    # print(D.labels_train[0:2])
    # print('\n\n')
    # print(D.sentences_test_raw[0:2])
    # print(D.sentences_test[0:2])
    # print(D.labels_test[0:2])

    #todo delete the following lines
    #print(D.sentences_test)
    # for x, y in zip(D.labels_test, D.labels_test_mapped):
    #     print(x, '\t', y)

    # create model
    model = get_nn_model(
        D.sentences_train_padded.shape[1],  # sentences_train_padded.shape[1]
        D.sentences_train_padded.shape[2],  # sentences_train_padded.shape[2]
        num_classes=10,
    )

    # train model
    train_model(
        model,
        D.sentences_train_padded,  # sentences_train_padded
        D.labels_train,
        D.sentences_val_padded,  # sentences_val_padded2
        D.labels_val,
        model_name="model_" + dataset,  # e.g. model_emp123 -> shows which dataset it was trained with
        dataset=dataset,
    )

    # evaluate model
    evaluation_report(model, D.sentences_test_padded, D.labels_test, dataset)
    # confusion matrix
    plot_confusion_matrix(model, D.sentences_test_padded, D.labels_test, dataset)
    # save classified sentences
    save_classified_test_sents(model, dataset, D.sentences_test_raw, D.sentences_test, D.sentences_test_padded, D.labels_test)

if __name__ == '__main__':
    main()