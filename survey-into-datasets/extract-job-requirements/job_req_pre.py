import pandas as pd
import tensorflow as tf
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re


# Load the dataset
job_data = pd.read_csv('data/Job_req.csv')

# Print the initial information about the dataset
print(job_data.info())

# Drop columns with all NaN values (which will include any unnamed columns like 'Unnamed: 8')
job_data = job_data.dropna(axis=1, how='all')

# Verify the columns are dropped
print(job_data.info())

# Fill NaN values in the 'Requirements' column with an empty string
job_data['Requirements'] = job_data['Requirements'].fillna('')

# Define a function to remove punctuation using TensorFlow
def remove_punctuation_tf(text_tensor):
    # Define the regular expression for punctuation
    regex_pattern = r'[{}]'.format(string.punctuation)
    # Replace punctuation with an empty string
    return tf.strings.regex_replace(text_tensor, regex_pattern, '')

# Define a function to convert text to lowercase using TensorFlow
def to_lower_case_tf(text_tensor):
    # Convert text to lowercase
    return tf.strings.lower(text_tensor)

# Convert the 'Requirements' column to a TensorFlow tensor
requirements_tensor = tf.convert_to_tensor(job_data['Requirements'].values, dtype=tf.string)

# Remove punctuation using the defined function
requirements_tensor = remove_punctuation_tf(requirements_tensor)

# Convert text to lowercase using the defined function
requirements_tensor = to_lower_case_tf(requirements_tensor)

# Convert the tensor back to a NumPy array and then to a DataFrame column
job_data['Requirements'] = requirements_tensor.numpy().astype(str)
# Join multi-line text into a single line
job_data['Requirements'] = job_data['Requirements'].apply(lambda x: " ".join(x.split('\n')))
# Verify the changes
print(job_data['Requirements'].head())

# Define English stopwords
english_stop_words = set(stopwords.words('english'))
english_stop_words.add('http')

# Define Indonesian stopwords using Sastrawi library
stopword_factory = StopWordRemoverFactory()
indonesian_stop_words = set(stopword_factory.get_stop_words())

# Combine English and Indonesian stopwords
all_stop_words = english_stop_words.union(indonesian_stop_words)
print(all_stop_words)
# Function to remove stopwords
def remove_stopwords(text):
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords
    filtered_words = [word for word in words if word.lower() not in all_stop_words]
    # Join the filtered words back into a string
    filtered_text = ' '.join(filtered_words)
    return filtered_text

# Apply stopwords removal to the 'Requirements' column
job_data['Requirements'] = job_data['Requirements'].apply(remove_stopwords)

# Verify the changes
print("\nremove stopwords actions:")
print(job_data['Requirements'].head())
# Define a function to remove extra spaces
def remove_extra_spaces(text):
    return re.sub(' +', ' ', text)
# Remove extra spaces
job_data['Requirements'] = job_data['Requirements'].apply(remove_extra_spaces)

# Function to lemmatize English words
def lemmatize_english(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

# Function to stem Indonesian words
def stem_indonesian(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(text)

# Apply lemmatization for English text
job_data['Requirements'] = job_data['Requirements'].apply(lemmatize_english)

# Apply stemming for Indonesian text
job_data['Requirements'] = job_data['Requirements'].apply(stem_indonesian)

# Verify the changes
print(job_data['Requirements'].head())