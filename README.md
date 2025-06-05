# NASA-APOD-Images-Classification

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

   

