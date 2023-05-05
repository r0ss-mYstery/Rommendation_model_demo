from dataloader import df
import jieba


def clean_text(text):
    # Check for missing or non-string values
    if not isinstance(text, str):
        return ''

    # Tokenize Chinese text with jieba
    tokens = list(jieba.cut(text, cut_all=False))

    # Remove punctuation and convert to lowercase
    cleaned_tokens = [token.lower() for token in tokens if token.isalnum()]

    # Join tokens with space
    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text


# Replace missing values with an empty string
# df['product_title'] = df['product_title'].fillna('')

# Apply the clean_text function to the product titles in the dataframe
df['cleaned_title'] = df['product_title'].apply(clean_text)


# Create a new dataframe with product titles and cleaned titles
cleaned_titles_df = df[['product_id', 'product_title', 'cleaned_title']]

# Save the cleaned titles to a new CSV file
output_csv_file = "./title/cleaned_titles_ch5.csv"
cleaned_titles_df.to_csv(output_csv_file, index=False)