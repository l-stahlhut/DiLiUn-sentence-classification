

# Sentence Classification

Code to preprocess and classify annotated sentences in student texts with a CNN.

Folder structure
```markdown
├── data
│   ├── raw
│   │   ├── emp_I1
│   │   │   ├── no_step.txt
│   │   │   ├── step1c.txt
│   │   │   ├── step1d.txt
│   │   │   ├── ...
│   │   │   ├── step3g.txt
│   │   ├── emp_I2
│   │   ├── ...
│   │   ├── syn_I2
├── preprocessing.py
├── nn
│   ├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```


## Preprocessing

Preprocess the data before training according to the flags, e.g.:
`$ python3 preprocessing.py data/raw --emp 1 2 3 --syn 1 2 --undersample`

this saves train/test/val data from empirically annotated data from iterations 1-3 and synthetic data from iterations 1&2. The training set is under sampled such that the classes are balanced.

Flags in the preprocessing script:

|                |Example Flag                         |Explanation                         |
|----------------|-------------------------------|-----------------------------|
|Empirical Data|`'--emp 1 2 3'`            |'Working with iterations 1, 2 and 3 from the empirical data'            |
|Synthetic Data          |`'--syn 1'`            |"Working with iteration 1 from the synthetic data."            |
|Undersampling          |`'--undersample'`|Undersample to the minority class?|



## Classification

The script trains the model and evaluates the performance.
`$ python3 nn/train_model.py --emp 1 2 3 --syn 1 2`


## TODO
- Modell trainieren, ev. verbessern
- preprocessing und classification in einem main Skript zusammenführen 