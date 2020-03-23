# A Deeper Look into Sarcastic Tweets Using Deep Convolutional Neural Networks

- [Paper](https://sentic.net/sarcasm-detection-with-deep-convolutional-neural-networks.pdf)
- [Dataset-Sentiment](https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset)
- [Dataset-Personality](https://drive.google.com/file/d/1xTg5iJZzzNEf3jJJKhBpgwkMnYDHaKQJ/view?usp=sharing)
- [Dataset-Reviews](https://github.com/ef2020/SarcasmAmazonReviewsCorpus)
- [GloVe pre-trained](http://nlp.stanford.edu/data/wordvecs/glove.6B.zip)

## Authors
- [Sanchit Ahuja](https://github.com/sanchit-ahuja)
- [Abhinav Tuli](https://github.com/AbhinavTuli)
- [Abhijeet Borole](https://github.com/abhijeetborole)

## Setup
- Create the environment
  - Modify ```prefix``` in environment file to the location where you wish to install the environment
  
  ```bash
  conda env create -f environment.yml
  conda activate sarcasm
  ```
- **Directory Structure** :
    - Sentiment
        - clean
        - og
        - ```clean.py```
        - models
    - Personality
        - utils
            - ```util_funcs.py```
        - datasets
            - ```clean_essays.pkl```
            - ```clean_essays.csv```
        - ```clean.py```
    - GloVe
        - 6B.300.dat (bcolz files for dealing with vectors)
        - ```6B.300_idx.pkl```
        - ```6B.300_workds.pkl```
        - ```glove_model.pkl```
        - ```glove_parse.py```
    - ```environment.yml```
    - README.md
    - `.gitignore`

You would need to download all the datasets and put them in their proper places. You would then need to run the clean scripts and generation scripts to get the preprocessed data.