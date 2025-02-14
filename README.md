
# News Classification Project

This project focuses on building a text classification model to categorize news articles into various categories based on headlines and short descriptions. The project leverages the BERT model for text embeddings and classification, and it is deployed using Google Colab and Streamlit on Google Cloud Platform (GCP).

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model](#model)
- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Deployment](#deployment)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
The aim of this project is to classify news articles into predefined categories. The project includes data preprocessing, model training, evaluation, and deployment.

## Google Drive Mount

![Alt Text](https://github.com/user-attachments/assets/de36af7a-8155-43ba-8cbb-7280cd40a1d8)


## Dataset
The dataset used in this project consists of news articles with the following columns:
- `REF_NO`: Reference number of the article
- `short_description`: Short description of the news article
- `category`: Category label for the news article

## Model
We use the BERT model for text classification. BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.

![Alt Text](https://github.com/user-attachments/assets/02888063-e095-433e-bb45-9902c98aab9f)

## Setup
To set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/news-classification.git
    ```

2. Navigate to the project directory:
    ```sh
    cd news-classification
    ```

3. Install the required libraries:
    ```sh
    pip install transformers torch pandas scikit-learn
    ```

4. Mount Google Drive in Google Colab:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

## Training
To train the model, follow these steps:

1. Load and preprocess the dataset.
2. Tokenize the text data using BERT tokenizer.
3. Split the data into training and validation sets.
4. Train the BERT model on the training set.


## Evaluation
Evaluate the model on the validation set to check its performance.

## Prediction
Use the trained model to make predictions on the test set and create a submission file.


![Alt Text](https://github.com/user-attachments/assets/76e52843-1d68-42ce-a408-cf42e76be071)

## Make predictions


## Deployment
Deploy the model using Streamlit on Google Cloud Platform (GCP).

1. Create a Streamlit app for real-time predictions.
2. Deploy the app on GCP.


## Results
The model achieved an accuracy of XX% on the validation set and XX% on the test set.

![Alt Text](https://github.com/user-attachments/assets/1a6bc06b-ab7b-47f2-9507-7cfa1e889a0f)


## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.






