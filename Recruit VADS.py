#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the trained model
model_filename = 'Recruit_VADS_model.pkl'

with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the candidate data
candidate_data_path = r"D:\1 ISB\Term 2\FP\FP project\Modifiedresumedata_data.csv"
candidate_data = pd.read_csv(candidate_data_path)

# Recreate the vectorizer during model loading
train_data_path = r"D:\1 ISB\Term 2\FP\FP project\Trainingdataset_data.csv"
train_data = pd.read_csv(train_data_path)
train_features = train_data[['sorted_skills', 'Certification', 'Experience']]
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(train_features.astype(str).agg(' '.join, axis=1))

# Streamlit UI
st.title("Recruit VADS - Candidate Relevancy Prediction")

job_title = st.text_input("Job Title")
skills = st.text_area("Skills")
experience = st.number_input("Experience", min_value=0, max_value=50)
certification = st.text_input("Certification")

apply_button = st.button("Apply")

if apply_button:
    # Create a vector for the user input
    user_input = pd.DataFrame({'sorted_skills': [skills], 'Certification': [certification], 'Experience': [experience]})
    user_vector = vectorizer.transform(user_input.astype(str).agg(' '.join, axis=1))

    # Calculate cosine similarity with each candidate
    candidate_vectors = vectorizer.transform(candidate_data[['sorted_skills', 'Certification', 'Experience']].astype(str).agg(' '.join, axis=1))
    similarity_scores = cosine_similarity(user_vector, candidate_vectors)[0]

    # Display results in a table
    result_df = pd.DataFrame({
        'Candidate Name': candidate_data['Candidate Name'],
        'Email ID': candidate_data['Email ID'],
        'Relevancy Score': similarity_scores
    })

    # Sort by relevancy score in descending order
    result_df = result_df.sort_values(by='Relevancy Score', ascending=False)

    # Display the result
    st.table(result_df[['Candidate Name', 'Email ID', 'Relevancy Score']].round(2).astype({'Relevancy Score': str} + '%'))

