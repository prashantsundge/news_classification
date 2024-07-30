
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

## Dataset
The dataset used in this project consists of news articles with the following columns:
- `REF_NO`: Reference number of the article
- `short_description`: Short description of the news article
- `category`: Category label for the news article

## Model
We use the BERT model for text classification. BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.

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

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

# Load the dataset
df = pd.read_csv('path_to_your_data.csv')

# Preprocess the data
# (Insert preprocessing steps here)

# Tokenize the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(df['short_description'].tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt')

# (Insert code for splitting data and training the model)
```

## Evaluation
Evaluate the model on the validation set to check its performance.

```python
# (Insert code for evaluating the model)
```

## Prediction
Use the trained model to make predictions on the test set and create a submission file.

```python
# Load the test data
test_df = pd.read_csv('/content/drive/My Drive/path_to_your_test_data.csv')

# Tokenize the test data
test_encodings = tokenizer(test_df['short_description'].tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt')

# Move the model to the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Make predictions
model.eval()
with torch.no_grad():
    input_ids = test_encodings['input_ids'].to(device)
    attention_mask = test_encodings['attention_mask'].to(device)
    
    outputs = model(input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Convert predictions to a list
predicted_labels = predictions.cpu().numpy().tolist()

# Add predictions to the test DataFrame
test_df['predicted_category'] = predicted_labels

# Rename the 'predicted_category' column to 'category'
submission_df = test_df[['REF_NO', 'predicted_category']].rename(columns={'predicted_category': 'category'})

# Save the submission file
submission_df.to_csv('/content/drive/My Drive/submission.csv', index=False, header=False)
```

## Deployment
Deploy the model using Streamlit on Google Cloud Platform (GCP).

1. Create a Streamlit app for real-time predictions.
2. Deploy the app on GCP.

```python
# (Insert Streamlit app code here)
```

## Results
The model achieved an accuracy of XX% on the validation set and XX% on the test set.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```




![image](https://github.com/user-attachments/assets/1a6bc06b-ab7b-47f2-9507-7cfa1e889a0f)
