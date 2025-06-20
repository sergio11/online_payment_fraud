# 💳 Fraud Detection with Deep Neural Networks (DNN) 🧠⚠️

As part of my journey through a **Deep Learning course**, I embarked on this project to explore one of the most impactful and challenging areas in applied AI: **fraud detection in financial transactions**. 💰🔍 This wasn’t just an exercise in model building — it was an opportunity to learn how **neural networks** can be used to identify rare, high-risk behaviors in real-world, imbalanced datasets. 

In fraud detection, the stakes are high: **fraudulent transactions often make up less than 1%** of the data, making the problem incredibly difficult to tackle with traditional classifiers. 🚨 This imbalance presented a fascinating challenge — and the perfect chance to apply deep learning techniques in a meaningful way.

Throughout the project, I performed detailed **Exploratory Data Analysis (EDA)** and applied **preprocessing techniques** such as normalization and resampling to ensure the data was ready for training. Then, I designed a **custom Deep Neural Network architecture** optimized for binary classification, experimenting with **activation functions**, **dropout**, **loss functions**, and various **optimization strategies**. The process taught me a great deal about **model regularization**, **handling overfitting**, and how to evaluate models beyond just accuracy — using metrics like **precision**, **recall**, and **F1-score** to better understand performance in an imbalanced context. 📊🧪

A special **thank you** goes out to [**Kavya's EDA notebook on Kaggle**](https://www.kaggle.com/code/kavya2099/online-payment-fraud-detection-eda) 🙏 — her excellent work provided a valuable reference point, helping me frame the problem and gain a deeper understanding of the dataset before diving into model development.

This project was more than a technical assignment — it was a hands-on lesson in **trust, security, and the power of data**. 🛡️💡 It reinforced the importance of responsible AI and gave me the tools to build solutions that matter.

<p align="center">
   <img src="https://img.shields.io/badge/pypi-3775A9?style=for-the-badge&logo=pypi&logoColor=white" />
   <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
   <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white" />
   <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
   <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" />
   <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" />
   <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" />
</p>

🙏 I would like to extend my heartfelt gratitude to [Santiago Hernández, an expert in Cybersecurity and Artificial Intelligence](https://www.udemy.com/user/shramos/). His incredible course on Deep Learning, available at Udemy, was instrumental in shaping the development of this project. The insights and techniques learned from his course were crucial in crafting the neural network architecture used in this classifier.

We would like to express our gratitude to **Jainil Shah** for creating and sharing the **Online Payment Fraud Detection dataset** on Kaggle. This dataset, which contains detailed historical information about fraudulent transactions, has been invaluable in building and training the machine learning model for detecting fraud in online payments. 

🌟 The dataset can be found on [Kaggle](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection/data). Your contribution is greatly appreciated! 🙌

## ⚠️ Disclaimer  
**This project was developed for educational and research purposes only.** It is an experimental implementation of **deep learning techniques for fraud detection** and should not be used in production systems or real-world financial applications.  

The model presented in this repository **has not been audited for regulatory compliance, financial security, or operational robustness**. Fraud detection in real financial environments requires rigorous testing, domain expertise, and compliance with legal and ethical standards.  

**Users should not rely on this project for real-time fraud prevention or financial decision-making.** Always consult industry professionals and use verified fraud detection solutions in real-world applications.  

## 🌟 Explore My Other Cutting-Edge AI Projects! 🌟

If you found this project intriguing, I invite you to check out my other AI and machine learning initiatives, where I tackle real-world challenges across various domains:

+ [🌍 Advanced Classification of Disaster-Related Tweets Using Deep Learning 🚨](https://github.com/sergio11/disasters_prediction)  
Uncover how social media responds to crises in real time using **deep learning** to classify tweets related to disasters.

+ [📰 Fighting Misinformation: Source-Based Fake News Classification 🕵️‍♂️](https://github.com/sergio11/fake_news_classifier)  
Combat misinformation by classifying news articles as real or fake based on their source using **machine learning** techniques.

+ [🛡️ IoT Network Malware Classifier with Deep Learning Neural Network Architecture 🚀](https://github.com/sergio11/iot_network_malware_classifier)  
Detect malware in IoT network traffic using **Deep Learning Neural Networks**, offering proactive cybersecurity solutions.

+ [📧 Spam Email Classification using LSTM 🤖](https://github.com/sergio11/spam_email_classifier_lstm)  
Classify emails as spam or legitimate using a **Bi-directional LSTM** model, implementing NLP techniques like tokenization and stopword removal.

+ [💳 Fraud Detection Model with Deep Neural Networks (DNN)](https://github.com/sergio11/online_payment_fraud)  
Detect fraudulent transactions in financial data with **Deep Neural Networks**, addressing imbalanced datasets and offering scalable solutions.

+ [🧠🚀 AI-Powered Brain Tumor Classification](https://github.com/sergio11/brain_tumor_classification_cnn)  
Classify brain tumors from MRI scans using **Deep Learning**, CNNs, and Transfer Learning for fast and accurate diagnostics.

+ [📊💉 Predicting Diabetes Diagnosis Using Machine Learning](https://github.com/sergio11/diabetes_prediction_ml)  
Create a machine learning model to predict the likelihood of diabetes using medical data, helping with early diagnosis.

+ [🚀🔍 LLM Fine-Tuning and Evaluation](https://github.com/sergio11/llm_finetuning_and_evaluation)  
Fine-tune large language models like **FLAN-T5**, **TinyLLAMA**, and **Aguila7B** for various NLP tasks, including summarization and question answering.

+ [📰 Headline Generation Models: LSTM vs. Transformers](https://github.com/sergio11/headline_generation_lstm_transformers)  
Compare **LSTM** and **Transformer** models for generating contextually relevant headlines, leveraging their strengths in sequence modeling.

+ [🩺💻 Breast Cancer Diagnosis with MLP](https://github.com/sergio11/breast_cancer_diagnosis_mlp)  
Automate breast cancer diagnosis using a **Multi-Layer Perceptron (MLP)** model to classify tumors as benign or malignant based on biopsy data.

+ [Deep Learning for Safer Roads 🚗 Exploring CNN-Based and YOLOv11 Driver Drowsiness Detection 💤](https://github.com/sergio11/safedrive_drowsiness_detection)
Comparing driver drowsiness detection with CNN + MobileNetV2 vs YOLOv11 for real-time accuracy and efficiency 🧠🚗. Exploring both deep learning models to prevent fatigue-related accidents 😴💡.


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

## 📊 Model Evaluation: Confusion Matrix & Classification Report

### **Confusion Matrix:**
The confusion matrix provides a detailed breakdown of the model's predictions, showcasing the model's performance in distinguishing between fraud and non-fraud transactions.

The confusion matrix for the model's predictions is as follows:

```python
[[593518 41923] [ 14886 620555]]
```

**Explanation:**
- **True Negatives (TN)**: 593,518 – Non-fraud transactions correctly identified as non-fraud.
- **False Positives (FP)**: 41,923 – Non-fraud transactions incorrectly classified as fraud.
- **False Negatives (FN)**: 14,886 – Fraud transactions incorrectly classified as non-fraud.
- **True Positives (TP)**: 620,555 – Fraud transactions correctly identified as fraud.

### **Classification Report:**
The classification report provides key metrics for evaluating the model's performance across both classes (fraud and non-fraud).

- **Precision (Fraud)**: 0.9367 – The model correctly identified 93.67% of the fraudulent transactions.
- **Recall (Fraud)**: 0.9766 – The model correctly detected 97.66% of all fraudulent transactions.
- **F1-score (Fraud)**: 0.9562 – The harmonic mean of precision and recall, reflecting strong model performance.
  
- **Precision (Non-Fraud)**: 0.9755 – The model correctly identified 97.55% of non-fraudulent transactions.
- **Recall (Non-Fraud)**: 0.9340 – The model correctly detected 93.40% of non-fraudulent transactions.
- **F1-score (Non-Fraud)**: 0.9543 – A balanced score for non-fraud detection.

**Overall Performance:**
- **Accuracy**: 0.9553 – The model achieved an accuracy of 95.53% on the test set.
- **Macro Average**:
  - **Precision**: 0.9561
  - **Recall**: 0.9553
  - **F1-score**: 0.9553
- **Weighted Average**:
  - **Precision**: 0.9561
  - **Recall**: 0.9553
  - **F1-score**: 0.9553

### **Key Insights from Evaluation:**
1. **High Recall for Fraud Detection (Class 1.0)**: The model performs exceptionally well at detecting fraud, capturing 97.66% of fraudulent transactions with high precision (93.67%).
   
2. **Good Performance for Non-Fraud (Class 0.0)**: While the recall for non-fraud is slightly lower (93.40%), the precision remains high (97.55%), indicating that the model is generally accurate in identifying non-fraudulent transactions as well.

3. **Class Imbalance Handling**: Despite the significant class imbalance (fraud transactions representing only 0.13% of total transactions), the model successfully managed the imbalance using **SMOTE** and **class weights**, ensuring that fraud detection was effective without overfitting the non-fraud class.

4. **Error Analysis**: There are still some false positives (non-fraud transactions incorrectly predicted as fraud) and false negatives (fraud transactions missed by the model), but these errors are relatively low, suggesting that the model performs well for both classes.

## 🔮 **Conclusion:**
- The model is highly effective at identifying fraudulent transactions while maintaining a strong balance between precision and recall for both fraud and non-fraud transactions.
- The **DNN model** has shown its capability to handle complex patterns in fraud detection, especially with the help of **SMOTE** for balancing the dataset and **class weighting** during training.

## 📚 Requirements

- Python 3.x
- TensorFlow (Keras)
- scikit-learn
- pandas
- imbalanced-learn
- matplotlib
- seaborn

## ⚠️ Disclaimer  
**This project was developed for educational and research purposes only.** It is an experimental implementation of **deep learning techniques for fraud detection** and should not be used in production systems or real-world financial applications.  

The model presented in this repository **has not been audited for regulatory compliance, financial security, or operational robustness**. Fraud detection in real financial environments requires rigorous testing, domain expertise, and compliance with legal and ethical standards.  

**Users should not rely on this project for real-time fraud prevention or financial decision-making.** Always consult industry professionals and use verified fraud detection solutions in real-world applications.  

## **🙏 Acknowledgments**

We would like to express our gratitude to **Jainil Shah** for creating and sharing the **Online Payment Fraud Detection dataset** on Kaggle. This dataset, which contains detailed historical information about fraudulent transactions, has been invaluable in building and training the machine learning model for detecting fraud in online payments. 

Thanks to this comprehensive dataset, we were able to explore key features and gain valuable insights into the patterns that distinguish fraudulent transactions from legitimate ones. We highly appreciate Jainil Shah's contribution to the data science community by providing this resource for further research and development in fraud detection.
  
A huge **thank you** to **jainilcoder** for providing the dataset that made this project possible! 🌟 The dataset can be found on [Kaggle](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection/data). Your contribution is greatly appreciated! 🙌

🙏 I would like to extend my heartfelt gratitude to [Santiago Hernández, an expert in Cybersecurity and Artificial Intelligence](https://www.udemy.com/user/shramos/). His incredible course on Deep Learning, available at Udemy, was instrumental in shaping the development of this project. The insights and techniques learned from his course were crucial in crafting the neural network architecture used in this classifier.

References
* [https://www.kaggle.com/code/kavya2099/online-payment-fraud-detection-eda](https://www.kaggle.com/code/kavya2099/online-payment-fraud-detection-eda)
* [https://www.pluralsight.com/guides/cleaning-up-data-from-outliers](https://www.pluralsight.com/guides/cleaning-up-data-from-outliers)
* [analyticsvidhya.com/blog/2021/05/detecting-and-treating-outliers-treating-the-odd-one-out/](analyticsvidhya.com/blog/2021/05/detecting-and-treating-outliers-treating-the-odd-one-out/)
* [https://www.analyticsvidhya.com/blog/2020/07/univariate-analysis-visualization-with-illustrations-in-python/](https://www.analyticsvidhya.com/blog/2020/07/univariate-analysis-visualization-with-illustrations-in-python/)
* [https://jovian.ai/aakashns-6l3/us-accidents-analysis](https://jovian.ai/aakashns-6l3/us-accidents-analysis)
* [https://thinkingneuron.com/how-to-visualize-the-relationship-between-two-categorical-variables-in-python/](https://thinkingneuron.com/how-to-visualize-the-relationship-between-two-categorical-variables-in-python/)


## Visitors Count

<img width="auto" src="https://profile-counter.glitch.me/online_payment_fraud/count.svg" />

## License ⚖️

This project is licensed under the MIT License, an open-source software license that allows developers to freely use, copy, modify, and distribute the software. 🛠️ This includes use in both personal and commercial projects, with the only requirement being that the original copyright notice is retained. 📄

Please note the following limitations:

- The software is provided "as is", without any warranties, express or implied. 🚫🛡️
- If you distribute the software, whether in original or modified form, you must include the original copyright notice and license. 📑
- The license allows for commercial use, but you cannot claim ownership over the software itself. 🏷️

The goal of this license is to maximize freedom for developers while maintaining recognition for the original creators.

```
MIT License

Copyright (c) 2024 Dream software - Sergio Sánchez 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
``
