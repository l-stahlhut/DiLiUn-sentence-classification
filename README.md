
# Sentence Classification

Code to preprocess and classify annotated sentences in student texts with a CNN.

Folder structure
```markdown
├── data
│   └── raw
│       ├── emp_I1
│       │   ├── no_step.txt
│       │   ├── step1c.txt
│       │   ├── step1d.txt
│       │   ├── ...
│       │   └── step3g.txt
│       ├── emp_I2
│       ├── ...
│       └── syn_I2
├── preprocessing.py
├── main.py
├── nn
│   └── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to use the script
Preprocess data specified in the command line arguments and train and evaluate the model with that data.
E.g. the following prompt specifies that empirically annotated data from the first four iterations and synthetically
generated data (ChatGPT) from the first two iterations will be used and the data will be undersampled to the
size of the minority class:
`$ python3 main.py data/raw --emp 1 2 3 4 --syn 1 2 --undersample`
The following promt would just use data from the first three iterations of empirically annotated data without
undersampling:
`$ python3 main.py data/raw --emp 1 2 3`

Flags in the preprocessing script:

|                |Example Flag                         |Explanation                         |
|----------------|-------------------------------|-----------------------------|
|Empirical Data|`'--emp 1 2 3'`            |'Working with iterations 1, 2 and 3 from the empirical data'            |
|Synthetic Data          |`'--syn 1'`            |"Working with iteration 1 from the synthetic data."            |
|Undersampling          |`'--undersample'`|Undersample to the minority class?|


## What this script does
Preprocessing: Removing stop words, named entities.
Classification: 10-way classification using a CNN.



## TODO
- Modell trainieren, ev. verbessern
- Kreativeres Preprocessing