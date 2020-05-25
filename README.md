# A Deeper Look into Sarcastic Tweets Using Deep Convolutional Neural Networks

- [Paper](https://sentic.net/sarcasm-detection-with-deep-convolutional-neural-networks.pdf)
- [Dataset-Sentiment](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [Dataset-Personality](https://drive.google.com/file/d/1xTg5iJZzzNEf3jJJKhBpgwkMnYDHaKQJ/view?usp=sharing)
- [Dataset-Reviews](https://github.com/ef2020/SarcasmAmazonReviewsCorpus)
<!-- - [GloVe pre-trained](http://nlp.stanford.edu/data/wordvecs/glove.6B.zip) -->

## Authors
- [Sanchit Ahuja](https://github.com/sanchit-ahuja)
- [Abhinav Tuli](https://github.com/AbhinavTuli)
- [Abhijeet Borole](https://github.com/abhijeetborole)


## Objectives and Implementation
- Read the paper thoroughly and understood all the necessary requiremnents to apply the paper to our datasets.
- We needed to first implement two separate models and use these pre-trained models to train our final model using CNN-SVM classifier.
- The pre-trained model involved training a sentiment classifier and personality classifier using CNNs on word embeddings.
- We used the pre-trained 300 dimension GloVe embeddings.

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

## Training the data
- You can train individual notebooks for sarcasm and personality datasets. 
- For personality, you will need to train for all the emotions individually. All the cleaned and separated emotions are there inside the Personality directory.
- You will need to change the PATH variable to your local path. You will also need to change the csv file names for loading into the dataloader in the notebook.
- We have provided all the pre-trained models in a drive link and its better to use them directly than training the model again.
- For sentiment, you can directly run the cells and train the model. The dataset was already available in the torchtext module.
- (CNN-SVM INSTRUCTIONS HERE BOROLE and apni notebook ka likh de ki wohi use karein)

## Results
## 1. Sentiment
We used the IMDB dataset for binary sentiment classification consisting of 50000 highly polar movie reviews classified as either Positive or Negative.
The train, validation and test split was 17500, 7500 and 25000 respectively.
We tried 2 different architectures:-
### First Model
#### Model Architecture
<img src="./Sentiment/first model architecture.jpeg" alt="d" width="800"/>

#### Train and Validation results
<img src="./Sentiment/Results/Old model Train and Validation Accuracy.png" alt="d" width="400"/>

#### Test results
<img src="./Sentiment/Results/Old Model Test Accuracy.png" alt="d" width="400"/>

### New Model

#### Model Architecture
<img src="./Sentiment/new model architecture.jpeg" alt="d" width="800"/>

#### Train and Validation results
<img src="./Sentiment/Results/Train and Validation Accuracy.png" alt="d" width="400"/>

#### Test results
<img src="./Sentiment/Results/Test Accuracy.png" alt="d" width="400"/>

## 2. Personality
- The dataset that we used to classifying personality is called OCEAN dataset. It consists of five personalities majorly. \
    1. OPN - [O] Openness to experience. (inventive/curious vs. consistent/cautious)
    2. CON - [C] Conscientiousness. (efficient/organized vs. easy-going/careless)
    3. EXT - [E] Extroversion. (outgoing/energetic vs. solitary/reserved)
    4. AGR - [A] Agreeableness. (friendly/compassionate vs. challenging/detached)
    5. NER- [N] Neuroticism. (sensitive/nervous vs. secure/confident)
- For personality, we had run couple of experiments dabbling with sending the complete text as well as chunking the text into sentences and then sending them to the network. We observed that chunking the text into sentences performed slightly better than sending the complete text. 
- For our experiment above we experimented with varied number of fully connected layer at the end. We tabulate our results below for both experiments down below.
### Complete text
- This experiment was done on a single personality and we extrapolated the results for other personalites as well.


| Fully connected layers       | Train acc.      | Val Acc  |
| ------------- |:-------------:| -----:|
| 70      | 92.05 | 58.32 |
| 80      | 99.85 |   62.30 |
| 90 | 98.19      |    58.50
| 100 |    99.95  |  58.84 |
| 110 |   99.55   |  60.48 |
| 120 |   89.06   |  59.45 |
| 130 |   99.50   |  58.47 |
| 140 |   99.75   |  58.27 |
| 150 | 99.95      |  62.15 |

### Text chunked into sentences

| Fully connected layers       | Train acc.      | Val Acc  |
| ------------- |:-------------:| -----:|
| 80      | 86.63 |   62.41 |
| 90 | 92.04|    59.91|
| 100 |    99.19  |  60.20 |
| 110 |   98.89  |  60.37 |
| 120 |   96.10   |  60.94 |
| 130 |   94.90   |  61.00 |
| 140 |   99.75   |  60.92 |
| 150 | 99.95      |  62.04 |

- We can observe that the chunked text converged significantly faster to a good accuracy as compared to the normal text. We also observed that the model with 80 fully connected layers worked the best.
 - The above experiments were ran on OPN personality.

### The best results for all the 5 personalities are tabulated below.

| Personality       | Train acc.      | Val Acc  |
| ------------- |:-------------:| -----:|
| OPN      | 86.63 |   62.41 |
| NEU | 99.75|    57.55|
| EXT |    99.65  |  55.96 |
| CON |   98.89  |  52.19 |
| AGR |   99.14   |  52.65 |


## Limitations
- The major limitation is that the dataset for Personality is too small. As a result we had a significant amount of overfitting in our results. 

## 3.Sarcasm
- The dataset that we used to classifying sarcasm is the imbalanced Amazon Review dataset which consists of ironic and regular reviews. 
- For sarcasm, we experimented with several classifiers such as MLP, RF, KNN and Logistic Regression and ultimately SVM worked best for the use case with 88.09%.

### The best results are shown below.

<!-- ### This repository is WIP. Things might not work as they intend to. Please wait for this message to be removed. -->
