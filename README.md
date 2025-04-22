# **Fraudulent Claim Detection**

## **Project Overview**

This project focuses on **fraud detection** in insurance claims using historical data and machine learning techniques. The goal is to predict fraudulent claims by analyzing features like **claim amounts**, **incident types**, **customer profiles**, and **claim severity**.

### **Key Features:**
- **Data Preprocessing**: Cleaning and preparing the data for modeling.
- **Feature Engineering**: Creating new features to improve model performance.
- **Model Training**: Building and evaluating different models like **Logistic Regression** and **Random Forest**.
- **Hyperparameter Tuning**: Fine-tuning the model for better accuracy and performance.
- **Model Evaluation**: Using various metrics like **ROC-AUC**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrix** to assess model performance.

---

## **Project Structure**

```
/fraudulent-claim-detection/
│
├── data/                  # Raw and processed data files
│   ├── train.csv          # Training data
│   ├── test.csv           # Test data
│
├── notebooks/             # Jupyter notebooks for exploratory analysis and model building
│   ├── EDA_and_Preprocessing.ipynb
│   ├── Model_Training_and_Evaluation.ipynb
│
├── src/                   # Python scripts for data processing and model training
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│
├── outputs/               # Generated outputs and model artifacts
│   ├── confusion_matrices/   # Folder with confusion matrices for each model
│   ├── performance_metrics/  # Folder with performance metrics (ROC AUC, F1-Score, etc.)
│
└── README.md              # This file
```

---

## **Setup and Installation**

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/fraudulent-claim-detection.git
   cd fraudulent-claim-detection
   ```

2. Install dependencies:

   It's recommended to use a virtual environment for managing dependencies. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

   Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   The **`requirements.txt`** file should contain the following libraries:

   ```
   pandas
   numpy
   matplotlib
   seaborn
   scikit-learn
   statsmodels
   ```

---

## **Usage**

### **1. Data Preprocessing**

The data preprocessing steps can be found in the **`notebooks/EDA_and_Preprocessing.ipynb`** notebook. It covers data cleaning, handling missing values, feature encoding, and scaling.

### **2. Model Training and Evaluation**

Run the **`Model_Training_and_Evaluation.ipynb`** notebook to train and evaluate the **Logistic Regression** and **Random Forest** models. The model evaluation includes various metrics like **accuracy**, **sensitivity**, **specificity**, **precision**, and **recall**.

### **3. Hyperparameter Tuning**

Use the **RandomizedSearchCV** or **GridSearchCV** in the **`model_training.py`** script for hyperparameter tuning. The best parameters are saved and can be used for further training.

---

## **Model Evaluation Metrics**

- **Accuracy**: Measures the proportion of correct predictions.
- **Sensitivity (Recall)**: The model’s ability to detect fraudulent claims (True Positives).
- **Specificity**: The model’s ability to correctly identify legitimate claims (True Negatives).
- **Precision**: The proportion of predicted fraudulent claims that are actually fraudulent.
- **F1-Score**: The harmonic mean of precision and recall, used to balance the two.

---

## **Conclusion**

- The **Random Forest** model achieved superior performance with an **ROC AUC of 0.90**, showing its ability to effectively differentiate between fraudulent and non-fraudulent claims.
- **Logistic Regression**, while slightly behind with an **ROC AUC of 0.79**, demonstrated high precision, making it suitable for scenarios where minimizing false positives is critical.
- Further **hyperparameter tuning** and model adjustments can be made to optimize the detection of fraudulent claims.

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### **Contact**

If you have any questions or suggestions, feel free to open an issue or contact me at [your-email@example.com].

---

This **README** provides an overview of the project, including instructions for setup, model training, evaluation, and conclusions. Let me know if you need any more additions or modifications!

## Contact
Created by [@[niranjansingh] and @[nitishnarayanan002]

