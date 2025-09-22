# E-commerce-Recommendation-Engine-An-Amazon-Data-Case-Study
A content-based recommendation system built on Amazon's robotics product data using TF-IDF and Cosine Similarity.

## Project Overview
This project is a hands-on implementation of a content-based recommendation system. It focuses on solving a real-world business problem: how to suggest relevant products to customers based on item similarity. The entire process, from raw data to a working model, is documented in this repository.

## Dataset
The dataset used for this project is a collection of Amazon e-commerce data for robotics-related products, scraped from Amazon.in. It includes product descriptions, prices, and customer ratings.

## Methodology
1. Data Cleaning & Preprocessing: The raw scraped data was cleaned to handle missing values and convert text-based ratings into a numerical format.
2. Feature Extraction: Natural Language Processing (NLP) techniques were applied to convert product descriptions into a numerical representation. Specifically, the TF-IDF (Term Frequency-Inverse Document Frequency) method was used to vectorize the text data.
3. Modeling: The core of the recommendation system was built by calculating Cosine Similarity between all products based on their TF-IDF vectors.
4. Recommendation Engine: A Python function was developed to query the similarity matrix, providing top-N recommendations for any given product.

## Key Outcomes
- Developed an end-to-end recommendation system from unstructured data.

- Demonstrated proficiency in data cleaning, NLP, and machine learning modeling.

- The final model successfully provides relevant product recommendations based on content similarity.

## Technologies Used
- Python
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter/Colab

## How to Run the Project
Simply open the **Ecommerce Data Analysis and Recommendation System.ipynb** notebook in Google Colab or Jupyter, and run all the cells in order. The notebook contains all the code and explanations to reproduce the results.
