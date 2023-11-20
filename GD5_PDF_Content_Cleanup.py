# Libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import PyPDF2
from io import BytesIO
from tqdm import tqdm
import time
import csv
import sys
from random import randint

# Load data from CSV file into a DataFrame
pdf_content = pd.read_csv('pdf_data.csv')

# Load data from CSV file into a DataFrame
article_link_directory = pd.read_csv('article_link_directory.csv')

# Add Inside PDF Text to article_link_directory
article_link_directory['Inside PDF Text'] = pdf_content['Inside PDF Text']

# Create a new column "Text" in article_link_directory based on conditions
article_link_directory['Text'] = article_link_directory.apply(lambda row: row['Inside PDF Text'] if row['Outside PDF Text'] == 'NO CONTENT' else row['Outside PDF Text'], axis=1)

# Clean the text data in 'Inside PDF Text' and 'Text' and 'Outside PDF Text' columns by replacing newline characters
article_link_directory['Inside PDF Text'] = article_link_directory['Inside PDF Text'].str.replace('\n', ' ')
article_link_directory['Text'] = article_link_directory['Text'].str.replace('\n', ' ')
article_link_directory['Outside PDF Text'] = article_link_directory['Outside PDF Text'].str.replace('\n', ' ')

# Extract document date from article_link_directory['Publishing Info']
# List to store extracted dates
Date = []

# Iterate through the 'Publishing Dates' column and extract dates based on the condition
# "Updated" dates are disregarded because they only seem to occur with certain documents of lesser importance and don't override the pdf included in content...so the pdf data reflects the original date

for date_string in article_link_directory['Publishing Info']:
    if 'Publicerad' in date_string:
        start_index = date_string.find('Publicerad') + len('Publicerad')
        end_index = date_string.find('·', start_index)
        if end_index != -1:
            extracted_date = date_string[start_index:end_index].strip()
            Date.append(extracted_date)
        else:
            # Handle the case when there's no '·' after 'Publicerad'
            Date.append(None)
    else:
        # Handle the case when 'Publicerad' is not mentioned in the string
        Date.append(None)

# Add the 'Date' list as a new column in the DataFrame
article_link_directory['Date'] = Date

# Extract document types from article_link_directory['Publishing Info']
# List to store extracted document types
Document_Type = []

# Iterate through the 'Publishing Info' column and extract document types based on the condition
for date_string in article_link_directory['Publishing Info']:
    last_dot_index = date_string.rfind('·')  # Find the last instance of "·"
    if last_dot_index != -1:
        # Extract string after the last "·"
        extracted_text = date_string[last_dot_index + 1:].strip()
        if 'från' in extracted_text:
            # Extract string before "från"
            extracted_text = extracted_text[:extracted_text.find('från')].strip()
            Document_Type.append(extracted_text)
        else:
            Document_Type.append(None)  # If "från" is not found, append None
    else:
        # Handle the case when there's no "·" in the string
        Document_Type.append(None)

# Add the 'Document_Type' list as a new column in the DataFrame
article_link_directory['Document Type'] = Document_Type

# Extract document source from article_link_directory['Publishing Info']
# List to store extracted sources
Source = []

# Iterate through the 'Publishing Info' column and extract sources based on the condition
for date_string in article_link_directory['Publishing Info']:
    # Find the index of "från"
    from_index = date_string.find('från')
    if from_index != -1:
        # Extract string after "från" and add it to the Source list
        extracted_source = date_string[from_index + len('från'):].strip()
        Source.append(extracted_source)
    else:
        # If "från" is not found, append None to indicate no source
        Source.append(None)

# Add the 'Source' list as a new row in the DataFrame
article_link_directory['Source'] = Source

# Identify columns to drop
columns_to_drop = ['Publishing Info', 'Collected Links', 'Outside PDF Text', 'Inside PDF Text']

# Drop the columns
article_link_directory = article_link_directory.drop(columns=columns_to_drop)

# Replace 'Full Collected Links' with the actual column name in your DataFrame
article_link_directory['PDF Indicator'] = (article_link_directory['Full Collected Links'] != 'NO CONTENT').astype(int)

# Replace the column names with the actual names in your DataFrame
article_link_directory.rename(columns={'Full Collected Links': 'PDF Link', 'Content Links': 'Content Page'}, inplace=True)

# Specify new column order
column_order = ['Date', 'Document Type', 'Source', 'PDF Indicator', 'Text', 'Content Page', 'PDF Link']

# Set new column order
article_link_directory = article_link_directory[column_order]

# Download .csv of the data
article_link_directory.to_csv('unclassified_data.csv', index=True)