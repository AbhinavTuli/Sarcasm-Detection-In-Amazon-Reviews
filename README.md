# A Deeper Look into Sarcastic Tweets Using Deep Convolutional Neural Networks

- [Paper](https://sentic.net/sarcasm-detection-with-deep-convolutional-neural-networks.pdf)
- [Dataset-Sentiment](https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset)
- [Dataset-Personality](https://drive.google.com/file/d/1xTg5iJZzzNEf3jJJKhBpgwkMnYDHaKQJ/view?usp=sharing)
- [Dataset-Reviews](https://github.com/ef2020/SarcasmAmazonReviewsCorpus)
<!-- - [GloVe pre-trained](http://nlp.stanford.edu/data/wordvecs/glove.6B.zip) -->

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
        - Results
        - ```Sentiment_CNN.ipynb```
    - Personality
        - utils
            - ```util_funcs.py```
        - datasets
    - ```environment.yml```
    - Sarcasm
        - Sarcasm.ipynb
    - README.md
    - `.gitignore`

### This repository is WIP. Things might not work as they intend to. Please wait for this message to be removed.
