from dataloader import df
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')


def clean_text(text):
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]

    # Join the tokens back into a single string
    cleaned_text = ' '.join(stemmed_tokens)

    return cleaned_text


# Apply the clean_text function to the product titles in the dataframe
df['cleaned_title'] = df['product_title'].apply(clean_text)


# Create a new dataframe with product titles and cleaned titles
cleaned_titles_df = df[['product_title', 'cleaned_title']]

# Save the cleaned titles to a new CSV file
output_csv_file = "./title/cleaned_titles_en.csv"
cleaned_titles_df.to_csv(output_csv_file, index=False)
