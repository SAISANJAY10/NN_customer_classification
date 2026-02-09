# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="901" height="905" alt="Screenshot 2026-02-05 222929" src="https://github.com/user-attachments/assets/7678e627-ba60-4788-ac92-47942f844e69" />

## DESIGN STEPS

STEP 1: Import necessary libraries and load the dataset.

STEP 2:
Encode categorical variables and normalize numerical features.

STEP 3:
Split the dataset into training and testing subsets.

STEP 4:
Design a multi-layer neural network with appropriate activation functions.

STEP 5:
Train the model using an optimizer and loss function.

STEP 6:
Evaluate the model and generate a confusion matrix.

STEP 7:
Use the trained model to classify new data samples.

STEP 8:
Display the confusion matrix, classification report, and predictions.

## PROGRAM

### Name: SAI SANJAY R
### Register Number: 212223040178

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader

# Load Dataset
dataset = pd.read_csv('/customer.csv')
print("Dataset Preview:\n", dataset.head())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Neural Network
class PeopleClassifier(nn.Module):
    def __init__(self, input_size, classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,classes)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = PeopleClassifier(X_train.shape[1], len(encoder.classes_))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(100):
    for xb,yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out,yb)
        loss.backward()
        optimizer.step()

print("\nTraining Completed")

# Evaluation
model.eval()
with torch.no_grad():
    preds = torch.argmax(model(X_test), dim=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test,preds))

print("\nClassification Report:")
print(classification_report(y_test,preds,target_names=encoder.classes_,zero_division=0))
sample = X_test[0].unsqueeze(0)
with torch.no_grad():
    pred = model(sample)
    result = encoder.inverse_transform([torch.argmax(pred).item()])

print("\nSample Prediction:", result[0])
```

## Dataset Information

![WhatsApp Image 2026-02-05 at 11 01 03 PM](https://github.com/user-attachments/assets/92fa4a71-0205-4257-bf10-42f85ddd188c)


## OUTPUT

### Confusion Matrix

![WhatsApp Image 2026-02-05 at 11 00 17 PM](https://github.com/user-attachments/assets/072a13ae-4687-47e9-8ca4-5ab67af41045)

### Classification Report

![WhatsApp Image 2026-02-05 at 11 00 17 PM (1)](https://github.com/user-attachments/assets/8d35f119-9f91-442f-811f-940e59bcc35f)


### New Sample Data Prediction

![WhatsApp Image 2026-02-05 at 10 59 32 PM](https://github.com/user-attachments/assets/b3756ede-3c02-4939-a8ee-726b7f9bf494)


## RESULT
Thus, a neural network classification model for the given dataset as been created successfully.
