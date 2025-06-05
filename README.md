# NASA-APOD-Images-Classification (To be updated in the future for better results)

## GROUP MEMBERS AND THEIR CONTRIBUTIONS :
• Imane Sayd: Preprocessing


• Ayomiposi Adebayo: Hybrid Model Building


• Yağmur Senanur Eroğlu: Performance Measurement

## 1.Introduction
The objective of this project is to develop a deep learning model capable of classifying astronomical images from NASA's Astronomy Picture of the Day (APOD) dataset into 4 distinct categories: ‘galaxy’, ‘star’, ‘nebula’ and ‘other’. By combining convolutional neural networks (CNNs) (for image classification) and LSTM (for text classification), the model aims to accurately identify and categorize celestial objects, enhancing our understanding and organization of astronomical imagery.

## 2.Problem Statement
The primary goal is to classify APOD images into four categories: 'galaxy', 'nebula', 'star', and 'other'. This classification task is crucial for organizing vast astronomical datasets and facilitating efficient retrieval and analysis of celestial images.

## 3.Data Description
•	Source: NASA's Astronomy Picture of the Day (APOD) dataset, found on Kaggle: https://www.kaggle.com/datasets/thomasanquetil/nasa-astronomy-picture-of-the-day-apod-extended/data


•	Size: The dataset comprises a substantial number of images, each accompanied by descriptive text and metadata (total of 9941 rows are used in this project).


•	Data Split:


  o	Training Set: 70% of the total data; 6958 data for training

  
  o	Validation Set: 10% of the total data; 994 data for validation

  
  o	Test Set: 20% of the total data; 1989 data for testing

  


•	Preprocessing:


  o	Text Data: ‘explanation’ column  ‘clean_explanation’ new column added

  
       	Tokenization: Converting text descriptions into sequences of tokens.

   
       	Stop Word Removal: Eliminating common words that do not contribute to the model's learning.

       
       	Stemming: Reducing words to their root forms to unify similar terms.

       
       	Padding: Ensuring uniform sequence lengths for model input (maxlen=200)

       
       	One-hot-encoding: After extracting the labels based on the ‘clean_explanation’ column, this operation is done on the new resulted column ‘label’ values.

       
  o	Image Data: ‘media_URL’ column  ‘image_data’ new column added

  
       	Resizing: by adjusting all images to a consistent size (224x224 pixels) suitable for CNN input.

       
       	Normalization: by scaling pixel values to a range of [0,1] to facilitate model training.

       

### Displaying the First 5 Training Images
![image](https://github.com/user-attachments/assets/920a89a1-9f05-4de5-bfa8-9d32e4f0e718)
![image](https://github.com/user-attachments/assets/bf5722dd-e9c7-45db-b719-e2c1628a1360)
![image](https://github.com/user-attachments/assets/69ff06e9-55af-414c-8c2a-995c9a754639)
![image](https://github.com/user-attachments/assets/070b2f2b-26f4-41e3-bc91-12d73e9fb320)
![image](https://github.com/user-attachments/assets/134ce80a-edfb-4da6-86f6-4494139d8130)

### Displaying the First 5 Validation Images
![image](https://github.com/user-attachments/assets/072cefe3-f957-480b-9345-724e55da41a8)
![image](https://github.com/user-attachments/assets/5b287520-55f4-44b2-a630-5936ad292e6e)
![image](https://github.com/user-attachments/assets/02530c94-1833-4606-8e17-4a5bb65af22c)
![image](https://github.com/user-attachments/assets/307e8fca-a3d7-41fd-af6a-7c4d13c48e21)
![image](https://github.com/user-attachments/assets/4fe9dd74-172a-4f8b-be68-f1778e9e1e39)

### Displaying the First 5 Test Images
![image](https://github.com/user-attachments/assets/bd2ce053-3dd5-45a8-b99a-7420adafc7aa)
![image](https://github.com/user-attachments/assets/495fd868-e299-4343-bf28-f76ab8941df2)
![image](https://github.com/user-attachments/assets/49eefa3f-d4db-4c13-b7a6-4841c9c65066)
![image](https://github.com/user-attachments/assets/fa6c354e-98e7-465e-9e25-3201f7ee22b8)
![image](https://github.com/user-attachments/assets/e7bbd8ac-f463-4f73-88f7-956ec24c3672)

## 4. The Deep Learning Model Used in this Project
*Hybrid deep learning model


*Developed a dual input hybrid model combining:


•	CNN (Convolutional Neural Network) for feature image extraction.


•	LSTM (Long Short Term Memory) for processing tokenized textual description.


This model is designed to classify NASA apod data into four categories: galaxy, nebula, star and other. just as my teammate said above.

### DATA SETUP
Using an already preprocessed input from the previous task:


X_image_train, X_image_val, X_image_test (Which has already been normalized)


X_text_train, X_text_val, X_image_test (Tokenized and padded)


y_train, y_val, y_test.


### MODEL ARCHITECTURE
*CNN Branch:


•	Three convolutional layers with ReLU activation and max pooling.


•	A flattening layer followed by a dense layer.


*LSTM Branch:


•	An embedding layer (vocab_size=10000, embedding_dim=128)


•	An LSTM layer (units=64) to learn sequential patterns in text.


*Final Layers:


•	The CNN and LSTM outputs are concatenated.


•	Followed by dense layers and a softmax output layer for classification.


*Training Strategy


•	Loss: categorical_crossentropy


•	Optimizer: Adam


•	Metrics: Accuracy


•	Batch Size: 32


•	Epochs: 20 for the first two and 30 for the last one


•	Validation: Used X_image_val, X_text_val, and y_val.


(+)Regularization & Overfitting Prevention:


•	EarlyStopping may be added later to monitors validation loss and stops training if performance stops improving.


## 5. Evaluation and Results
During the training of the model both accuracy and loss values have been carefully monitored.


Accuracy - Train and Val Accuracy have increased over the time. Model has learned better with each epoch. As the graph shows , the model learned well during training but performed slightly worse in validation with a small difference.


Loss - The difference between the training and validation indicates that the model fits the training data well but at the same time there may be some slight overfitting in the validation data.

Below is the evaluation of the process : 
![image](https://github.com/user-attachments/assets/97cf32d5-b4f8-4880-8804-e8a11386b003)

# Test Accuracy
Model achieved an accuracy rate of %89.34 in the evaluation part. The result shows that the hybrid structure which can process both visual and description texts together effectively classifies APOD images.
The class-based classification results below indicate that the model generally performs well. In particular, the high precision, recall and F1 score values in the ‘nebula’ and ‘other’ classes shows that the model can also distinguish those classes with high accuracy. The results from ‘star’ class also at a satisfactory level, the model has learned the patterns related to this category successfully. However, the relatively low recall value -which is 0.60- for ‘the galaxy’ class points to a class imbalance, as the number of examples belonging to this class in the test set is quite small. This situation has led to the model being unable to make robust generalizations regarding this class. The overall accuracy has been calculated as ‘%89.34’ with a macro average F1-score of 0.85 and a weighted average F1-score of 0.89.

# Confusion Matrix
Overall, Confusion Matrix indicates that the model performs well in the ‘nebula’ and also ‘other’ classes. But particularly struggles with classes that have a low number of samples like ‘galaxy’. This situation suggests that future studies should address class imbalance or implement data augmentation techniques. 
![image](https://github.com/user-attachments/assets/00631771-7ba1-4e91-bb4d-75a8d12ac5f9)
![image](https://github.com/user-attachments/assets/700354e0-676e-4700-a6c0-9145bed52423)
![image](https://github.com/user-attachments/assets/f2425665-8e8d-4a98-bd9b-ed07c29893ad)

As a result of three fold cross validation processes conducted to evaluate the models generalization capacity, accuracy values of  ‘%91.25 , %87.28 , %90.00’ were obtained respectively. By taking the arithmetic mean of these values the cross validation accuracy calculated as %89.54. These results indicate that the model demonstrates consistent performance not only in a single test split but at the same time across different subsets of data. This situation suggests that the model has largely avoided the tendency of overfitting and maintains its generalizability.

## 6. Conclusion
On this project, the aim is to classify astronomical images into four categories using NASA’s ASTRONOMY PICTURE of the DAY dataset. For this purpose a hybrid model consisting of a combination of CONVOLUTIONAL NEURAL NETWORKS and LONG SHORT TERM MEMORY layers was developed. The model achieved an accuracy rate of %89.34 on test data and showed consistent results with an accuracy of 89.54 through three fold cross validation. High precision and recall values were obtained in the nebula and other classes, however the low success rate in the galaxy class was due to data scarcity.
The best result of the model was achieved using an early stopping mechanism. Data preprocessing steps conducted before training allowed the model to process the data more efficiently. Imbalances in class distribution and some content similarities between classes posed challenges during the models learning process. 

Hence, the developed hybrid CNN-LSTM model has presented an effective approach in multidimensional data classification tasks by successfully combining visual and textual components. In the future it may be possible to improve performance through the integration of data augmentation techniques and modern transformer-based models.








   

