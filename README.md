# Fraud Detection Model with Deep Neural Networks (DNN)

## Overview

This project involves building a **Fraud Detection** model using **Deep Neural Networks (DNN)**. The goal is to predict fraudulent transactions in a financial dataset. This project covers the entire process, from **Exploratory Data Analysis (EDA)** to data preprocessing, model architecture, and training.

## 🧑‍🔬 Exploratory Data Analysis (EDA)

During the **EDA** phase, the following insights were gathered from the dataset:

- **Fraud Percentage**: Fraud transactions account for only **0.13%** of the total transactions in the dataset, making it highly imbalanced.
- **Fraud by Transaction Type**: Fraud occurred mostly in **cashout** and **transfer** transaction types. Fraud was rare in **payment** types.
- **Fraud Flagging (isFlaggedFraud)**: Very few fraud transactions were flagged (only **16 out of 8,213 fraud transactions**), indicating a need for improved fraud detection.
- **Incorrectly Flagged Transactions**: **99.805%** of fraud transactions were incorrectly flagged as non-fraud, underlining a significant issue in fraud detection algorithms.
- **Fraud Transaction Amount Range**: Fraudulent transactions predominantly occurred in the range of **₹1.3 Lakh - ₹3.6 Lakh**, with the majority falling between ₹3.4 Lakh - ₹3.6 Lakh.

### Key Conclusions:
- **Targeted Fraudulent Amounts**: Focus on high-value transactions, particularly in the ₹1 - ₹4 Lakh range.
- **Fraud Mode**: Fraud is most prevalent in **cashout** and **transfer** modes, which should be prioritized in fraud prevention efforts.
- **Improvements Needed**: There is a significant gap in fraud detection, especially in the **flagging process**.

## 🧹 Data Preprocessing

The dataset required significant cleaning before being fed into the model:

### Steps taken:
1. **Removed Irrelevant Features**:
   - Removed columns like `nameDest` and `nameOrig` due to their high cardinality and low impact on fraud detection.
   
2. **Feature Engineering**:
   - Created new features, such as `balance_diff_org` and `balance_diff_dest`, by calculating the difference between the old and new balances for both the origin and destination accounts.

3. **Encoded Categorical Variables**:
   - Applied **One-Hot Encoding** to the `type` column to convert it into binary variables for the model to process.

4. **Scaled Numerical Features**:
   - Normalized and scaled features like `amount`, `balance_diff_org`, and `balance_diff_dest` using **MinMaxScaler** to ensure they are within a similar range.

## ⚖️ Handling Class Imbalance

Given that fraud accounts for only **0.13%** of the total transactions, handling the class imbalance was a crucial step. 

### Steps taken:
1. **SMOTE (Synthetic Minority Over-sampling Technique)**:
   - Applied SMOTE to oversample the minority class (fraud) and balance the dataset.
   - Ensured that both fraud and non-fraud classes had a similar number of instances for training.

2. **Class Weights**:
   - Used **class weighting** during model training to give higher importance to the minority class (fraud) so that the model doesn't get biased toward the majority class.

## 🧠 Neural Network Architecture

The final model architecture chosen was a **Deep Neural Network (DNN)**. The choice of DNN was based on the following factors:
- **Non-linear relationships**: Fraud detection involves complex patterns that linear models may struggle to capture.
- **Large dataset**: With a large dataset and multiple features, DNNs are effective at learning intricate patterns.
- **Feature Interactions**: DNNs are capable of learning interactions between features, which is crucial in fraud detection.

### Architecture:
- **Input Layer**: 20 nodes, one for each feature in the dataset.
- **Hidden Layers**:
  - **First Hidden Layer**: 64 nodes with ReLU activation to capture complex patterns.
  - **Second Hidden Layer**: 32 nodes to reduce dimensionality and focus on essential features.
  - **Third Hidden Layer**: 16 nodes for further abstraction.
- **Output Layer**: 1 node with a **sigmoid activation** function to predict the probability of fraud (1 for fraud, 0 for non-fraud).

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Initialize the Neural Network model
model = Sequential()

# Input Layer (First Hidden Layer)
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))

# Dropout layer to prevent overfitting
model.add(Dropout(0.2))

# Additional Hidden Layers
model.add(Dense(64, activation='relu'))  # Second Hidden Layer
model.add(Dropout(0.2))  # Dropout layer for regularization

model.add(Dense(32, activation='relu'))  # Third Hidden Layer
model.add(Dropout(0.2))  # Dropout layer for regularization

# Output Layer (for binary classification)
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model to verify architecture
model.summary()
```

## ⚡ Model Training with Early Stopping

To ensure optimal training and avoid overfitting, **EarlyStopping** was implemented.

### Key points:
- **Monitor**: We monitored the **validation loss** during training.
- **Patience**: Set to **10 epochs**, meaning the model would stop training if the validation loss didn't improve after 10 consecutive epochs.
- **Restoring Best Weights**: The model’s best weights were restored from the epoch where validation loss was the lowest, ensuring the model was in the best state at the end of training.

## 🔮 Conclusion

This project has successfully identified key insights from the dataset, cleaned and preprocessed the data, and built a robust **Deep Neural Network (DNN)** model for fraud detection.

### Key Takeaways:
- **Fraud detection improvements are needed** due to high misclassification rates and incorrect flagging.
- **Targeting high-value transactions** (₹1 - ₹4 Lakh) will help in detecting a significant portion of fraud.
- The **DNN model** performed well in learning complex patterns in the data, and **SMOTE** was crucial for handling the class imbalance.

### Next Steps:
- Further refine feature engineering techniques.
- Implement more advanced techniques for fraud detection, such as ensemble methods or reinforcement learning.
- Continuously monitor and retrain the model with new data to improve accuracy and reliability.

## 🚀 Future Work

- **Model Fine-Tuning**: Explore hyperparameter optimization techniques such as **Grid Search** or **Random Search** to improve model performance.
- **Real-Time Fraud Detection**: Deploy the model in real-time systems for live fraud detection on financial transactions.
- **Enhanced Feature Engineering**: Experiment with additional features like transaction time, geolocation, and other behavioral data to improve model accuracy.

## 📚 Requirements

- Python 3.x
- TensorFlow (Keras)
- scikit-learn
- pandas
- imbalanced-learn
- matplotlib
- seaborn
