import streamlit as st
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

def display_info():
    st.subheader("Naive Bayes - Overview")
    st.write("""
    Naive Bayes is a probabilistic classifier based on Bayes' Theorem. It assumes that the features are independent given the class.
    It is commonly used for classification problems like spam detection.
    """)

    st.subheader("Mathematics Behind Naive Bayes")
    st.write("""
    Naive Bayes uses **Bayes' Theorem** to calculate the posterior probability of a class given the observed features.
    \[
    P(C|X) = \frac{P(X|C) P(C)}{P(X)}
    \]
    """)

    st.subheader("Python Code Example")
    st.code("""
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# Example dataset
emails = ['Free money now!', 'Hello, how are you?', 'Get rich quick with this scheme', 'Meeting tomorrow at 10am']
labels = ['spam', 'ham', 'spam', 'ham']

# Vectorize text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Predictions and accuracy
y_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
""")
    st.write("""
    This example uses Naive Bayes to classify emails as **spam** or **ham** based on their content.
    """)

    st.subheader("Helpful Links")
    st.write("""
    - [Machine Learning Mastery - Naive Bayes](https://machinelearningmastery.com/naive-bayes-classifier-in-python/)
    - [Scikit-learn Documentation - Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
    """)

