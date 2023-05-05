import pandas as pd

# the path to your dataset file or database connection
data_file = "./title/iDeal_title.csv"
original_df = pd.read_csv(data_file)

# Keep only the 'title' column
df = original_df[['id', 'title']].rename(columns={'id': 'product_id', 'title': 'product_title'})
