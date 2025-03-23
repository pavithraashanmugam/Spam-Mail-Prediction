# Spam Mail Detection using Logistic Regression

## Overview
This project aims to detect whether an email is **Spam** or **Ham** (non-spam) based on its content using machine learning. The model is built using **Logistic Regression**, one of the simplest yet effective algorithms for binary classification problems. The approach is divided into several stages: data collection, data pre-processing, feature extraction, model training, evaluation, and prediction.

The goal is to classify emails into two categories:
- **Spam** (denoted by `0`)
- **Ham** (denoted by `1`)

## Key Concepts
- **Logistic Regression**: A popular algorithm used for binary classification tasks.
- **TF-IDF Vectorization**: A technique to convert text data into numerical data by measuring the frequency of words in a document relative to the entire dataset. This method helps capture the important features of the text for classification purposes.
- **Train-Test Split**: The dataset is divided into two parts, one for training the model and the other for evaluating its performance.
- **Accuracy Score**: A metric used to evaluate the model's performance. It is the ratio of correct predictions to total predictions.

## Project Workflow
1. **Data Collection and Pre-processing**:
   - The dataset is loaded from a CSV file that contains labeled data, with emails marked as either "ham" (non-spam) or "spam".
   - Empty data entries are handled and replaced with null strings.
   
2. **Label Encoding**:
   - The labels are converted from categorical values (`spam` and `ham`) to numerical values (`0` and `1`), where `0` denotes "Spam" and `1` denotes "Ham".

3. **Feature Extraction**:
   - The text data is converted into numerical data using **TF-IDF Vectorization**. This transformation helps the model to interpret the content of emails as numeric values based on word frequency.

4. **Model Training**:
   - The **Logistic Regression** model is trained using the training data (80% of the dataset). The model learns to predict whether an email is spam or ham based on its content.

5. **Model Evaluation**:
   - The model is tested using the test data (20% of the dataset), and the accuracy of the model is calculated using the `accuracy_score` function.
   - **Training Accuracy**: The model's accuracy on the training data.
   - **Testing Accuracy**: The model's accuracy on the test data.

6. **Prediction**:
   - The trained model is used to predict whether new email data is "Spam" or "Ham".

## Installation and Usage

### Requirements:
- Python 3.x
- Pandas
- NumPy
- scikit-learn
- Jupyter (for interactive environment)

You can install the necessary libraries by running:
```bash
pip install pandas numpy scikit-learn
```

### Steps:
1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Load the Dataset**:
   The dataset should be a CSV file (`mail_data.csv`) with two columns:
   - **Category**: Labels (`spam`, `ham`)
   - **Message**: Email content

3. **Run the Script**:
   - Open the script in a Jupyter Notebook or a Python environment.
   - Execute each section to follow the workflow: data loading, preprocessing, feature extraction, model training, evaluation, and prediction.

### Sample Input:
```plaintext
Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
```

### Output:
```plaintext
Mail 1 is HAM mail
Mail 2 is SPAM mail
```

## Accuracy Score:
- **Training Accuracy**: `96.77%`
- **Testing Accuracy**: `96.68%`

These accuracy scores indicate that the model performs well on both the training and testing datasets, with a slight difference that is expected due to the division of the data.

## Key Concepts and Libraries Used:
- **Pandas**: Used for data manipulation and analysis.
- **NumPy**: Used for numerical operations.
- **scikit-learn**: Contains essential machine learning libraries for model building and evaluation.
  - **train_test_split**: Splits the data into training and testing datasets.
  - **TfidfVectorizer**: Converts text data into numerical features using TF-IDF.
  - **LogisticRegression**: A classification algorithm used to predict spam or ham emails.
  - **accuracy_score**: A function used to evaluate the model’s performance.

## Conclusion:
This Spam Mail Detection system demonstrates a simple yet effective approach to classify emails as either spam or non-spam (ham). By applying **Logistic Regression** and **TF-IDF Vectorization**, we can achieve high accuracy and create a robust model for email classification tasks.

---
