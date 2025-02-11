# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle
# import re
# import nltk
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split

# nltk.download('stopwords')
# nltk.download('wordnet')

# # Load saved models
# with open('model/Email.pkl', 'rb') as file:
#     model = pickle.load(file)


# # Text preprocessing
# def preprocess_text(text):
#     stop_word = set(stopwords.words('english'))
#     stemmer = PorterStemmer()
#     lemmatizer = WordNetLemmatizer()
#     text = re.sub(r'[^\w\s]', '', text).lower()
#     tokenized_text = text.split()
#     tokens = [stemmer.stem(word) for word in tokenized_text if word not in stop_word]
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     return ' '.join(tokens)

# # Streamlit UI
# st.title("Email Classification with NLP")

# st.write("### Enter an email below to classify it as spam or not.")
# user_input = st.text_area("Enter the email content:")

# if st.button("Predict"):
#     if user_input:
#         processed_text = preprocess_text(user_input)
#         prediction = model.predict()
#         result = "Spam" if prediction == 1 else "Not Spam"
#         st.write(f"### Prediction: {result}")
#     else:
#         st.warning("Please enter text to classify.")




import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load both the model and vectorizer
with open('model/Email.pkl', 'rb') as file:
    model = pickle.load(file)
    
with open('model/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def preprocess_text(text):
    """Enhanced preprocessing for better accuracy"""
    stop_word = set(stopwords.words('english'))
    abuse_related_words = {'you', 'your', 'yourself', 'he', 'she', 'they', 'them'}
    stop_word = stop_word - abuse_related_words
    
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    # Convert to lowercase and handle special characters
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    text = re.sub(r'(.)\1+', r'\1\1', text)  # Handle repeated characters
    
    # Enhanced tokenization and cleaning
    tokenized_text = text.split()
    tokens = [stemmer.stem(word) for word in tokenized_text if word not in stop_word]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Streamlit UI
st.title("Email Content Analysis System")

st.markdown("""
    <style>
    .severe-abusive {
        color: #FF0000;
        font-weight: bold;
    }
    .moderate-abusive {
        color: #FFA500;
        font-weight: bold;
    }
    .non-abusive {
        color: #008000;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.write("### Enter email content for analysis")
user_input = st.text_area("Enter the email content:", height=200)

if st.button("Analyze Content"):
    if user_input:
        try:
            # Preprocess and predict
            processed_text = preprocess_text(user_input)
            text_features = vectorizer.transform([processed_text])
            prediction = model.predict(text_features)
            confidence = model.predict_proba(text_features)[0]
            
            # Determine severity based on confidence
            if prediction[0] == 1:
                if confidence[1] >= 0.8:
                    severity = "Non abusive"
                    style_class = "severe-abusive"
                else:
                    severity = "abusive"
                    style_class = "moderate-abusive"
                    
                st.markdown(f'<div class="prediction-box"><p class="{style_class}">⚠️ {severity} Content Warning</p></div>', unsafe_allow_html=True)
                
                st.write("### Analysis Details:")
                st.write(f"- Confidence Level: {confidence[1]:.2%}")
                st.write("- Category: Potentially Inappropriate Content")
                
                st.write("### Specific Concerns:")
                if severity == "Severe":
                    st.write("- High probability of harmful content")
                    st.write("- May contain strongly inappropriate language")
                    st.write("- Possible presence of aggressive or hostile content")
                else:
                    st.write("- Moderate concerns about content appropriateness")
                    st.write("- May contain mildly inappropriate language")
                    st.write("- Possible presence of controversial content")
                
            else:
                st.markdown('<div class="prediction-box"><p class="non-abusive">✅ Content appears appropriate</p></div>', unsafe_allow_html=True)
                st.write(f"Confidence Level: {confidence[0]:.2%}")
                st.write("No concerning content detected")
            
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.write("Please ensure proper model configuration.")
    else:
        st.warning("Please enter text to analyze.")

# Sidebar guidelines
st.sidebar.title("Analysis Guidelines")
st.sidebar.write("""
### We analyze for:
- Inappropriate language
- Controversial content
- Potentially harmful messaging
- Overall tone and context

### Severity Levels:
- 🔴 Severe: High-confidence detection
- 🟠 Moderate: Medium-confidence detection
- 🟢 Appropriate: No concerns detected

### Note:
This tool provides automated analysis. Final judgment should involve human review.
""")