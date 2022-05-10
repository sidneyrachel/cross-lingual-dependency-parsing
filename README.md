# Cross-lingual biaffine graph-based dependency parser

This repository provides the implementation of cross-lingual biaffine graph-based dependency parser which utilizes multilingual cased BERT.

How to run the program:
1. Set up a new conda environment with Python 3.
2. Download torchtext 0.9.0, torch, nltk, transformers, and matplotlib using pip command.
3. Download `ud-treebanks-v2.9` from [Universal Dependencies 2.9](http://hdl.handle.net/11234/1-4611) and place it inside `external_resources` folder.
4. Create `figures`, `logs`, and `trained_models` folders to store the results of the experiments.
5. Set the settings and parameters of the candidate model in `.json` file and store it inside the `configs` folder. You can find existing examples inside the folder.
6. Run the training using `python main.py <name of the config file>`, for example: `python main.py czech_serbian_few` if you store the config inside `configs/czech_serbian_few.json`.
