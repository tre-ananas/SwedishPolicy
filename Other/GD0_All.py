# GovDocsOperational.py
# 10/10/23

# DESCRIPTION
# Process the data scraped by GovDocs_Step1_Selenium

# 11/9/23: Sleep times increased slightly so as not to overload the archive

# END OF TITLE AND DESCRIPTION

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 1: LIBRARIES AND SETTINGS

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

# Other Settings
# pd.set_option('display.max.colwidth', None) # max display width

# END OF STEP 1

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 2: LOAD DATA

# Load data from CSV file into a DataFrame
csv_file_path = "publication_info.csv"  # Update with the correct file path
existing_data = pd.read_csv(csv_file_path)

# Rename the columns to match your desired names
existing_data.columns = ['Content Links', 'Publishing Dates']

# Append the loaded data to the existing DataFrame
article_link_directory = existing_data.copy()

# END OF STEP 2

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 3: COLLECT THE PDF LINKS FROM EACH CONTENT PAGE CONTAINING A PDF; IF A CONTENT PAGE CONTAINS NO PDF, GRAB ALL TEXT INSTEAD

# Initialize an empty list to collect the links
layered_links = []
text_data_list = []

# No content alert
no_content_alert = "NO CONTENT"

for link in tqdm(article_link_directory['Content Links'], desc="Step 1: Collecting PDF Links", unit="link"):
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(link)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, "html.parser")

            # Find the first instance of <ul class="list--Block--icons">
            ul_tag = soup.find('ul', class_='list--Block--icons')

            # Check if the <ul> tag is found
            if ul_tag:
                # Find the first <a> tag with href inside the <ul> tag
                first_link = ul_tag.find('a', href=True)
                # no_content_alert = "NO CONTENT"
                if first_link:
                    layered_links.append(first_link['href'])
                    text_data_list.append(no_content_alert)
                else:
                    layered_links.append(no_content_alert)
            else:
                # If <ul> tag is not found, extract text from <p> tags
                paragraphs = soup.find_all('p')
                text_data = '\n'.join([p.get_text(strip=True) for p in paragraphs])
                text_data_list.append(text_data)
                layered_links.append(no_content_alert)
        else:
            print(f"Failed to fetch URL: {link}, Status code: {response.status_code}")
            # If <ul> tag is not found, extract text from <p> tags
            paragraphs = soup.find_all('p')
            text_data = '\n'.join([p.get_text(strip=True) for p in paragraphs])
            text_data_list.append(text_data)
            layered_links.append(no_content_alert)

    except Exception as e:
        print(f"An error occurred while processing URL: {link}, Error: {str(e)}")
        text_data_list.append(no_content_alert)
        layered_links.append(no_content_alert)

    # Introduce a random delay time before the next request
    time.sleep(randint(4, 6))  # Adjust the delay time as needed

# Create a DataFrame with the lists and rename columns
additional_data = pd.DataFrame({'Collected Links': layered_links, 'Outside PDF Text': text_data_list})
# Concatenate the new DataFrame with the original DataFrame along the columns axis (axis=1)
article_link_directory = pd.concat([article_link_directory, additional_data], axis=1)

# Add prefix to the links in the result_df to complete the links
# Define the prefix to add
prefix = 'https://www.regeringen.se'

# Define a function to conditionally add the prefix
def add_prefix(link):
    if link == 'NO CONTENT':
        return link
    else:
        return f'{prefix}{link}'

# Use the .apply() method with the defined function to add the prefix conditionally
article_link_directory['Full Collected Links'] = article_link_directory['Collected Links'].apply(add_prefix)

# Remove the repetitive string in front of all the rows on the Outside PDF Text
# Define the string to remove
string_to_remove = "På regeringen.se använder vi kakor för att löpande förbättra webbplatsen. Du väljer själv om du accepterar kakor.Läs om kakor\nHuvudnavigering\nHuvudnavigering\n"

# Define a function to conditionally remove the specified string
def remove_string(text):
    if text == 'NO CONTENT':
        return text
    else:
        return text.replace(string_to_remove, '', 1)  # Remove the specified string only from the beginning

# Apply the defined function to 'Outside PDF Text' column
article_link_directory['Outside PDF Text'] = article_link_directory['Outside PDF Text'].apply(remove_string)

# Download article_link_directory as a fail safe
article_link_directory.to_csv('article_link_directory_save_step_2.csv', index=False)

# END OF STEP 3

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 4: COLLECT PDF TEXT CONTENT

# Initialize an empty list to collect text data from PDFs
pdf_text_data = []

for pdf_url in tqdm(article_link_directory['Full Collected Links'], desc = "Step 2: Collecting PDF Text", unit = "link"):
    
    # Check if the URL contains the NO CONTENT alert
    if "NO CONTENT" in pdf_url:
        # If yes, append "NO CONTENT" to pdf_text_data and continue to the next URL
        pdf_text_data.append("NO CONTENT")

    else:
        try:
            # Send an HTTP GET request to the PDF URL
            response = requests.get(pdf_url)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Wrap the response content in a BytesIO object
                pdf_bytes = BytesIO(response.content)

                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(pdf_bytes)

                # Initialize an empty string to store the text data
                text_data = ""

                # Use len(reader.pages) to determine the number of pages
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_data += page.extract_text()

                # Append the extracted text to the pdf_text_data list
                pdf_text_data.append(text_data)
            else:
                print(f"Failed to fetch PDF URL: {pdf_url}, Status code: {response.status_code}")
                pdf_text_data.append("NO CONTENT")

        except Exception as e:
            print(f"An error occurred while processing PDF URL: {pdf_url}, Error: {str(e)}")
            pdf_text_data.append("NO CONTENT")

    # Introduce a random delay time before the next request
    time.sleep(randint(4, 6))  # Adjust the delay time as needed


# Create a new column "Inside PDF Text" in article_link_directory and assign pdf_text_data to it
article_link_directory['Inside PDF Text'] = pdf_text_data

# Download article_link_directory as a fail safe
article_link_directory.to_csv('article_link_directory_save_step_3.csv', index=False)

# END OF STEP 4

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 5: COMBINE PDF AND TEXT DATA INTO A SINGLE COLUMN

# Create a new column "Text" in article_link_directory based on conditions
article_link_directory['Text'] = article_link_directory.apply(lambda row: row['Inside PDF Text'] if row['Outside PDF Text'] == 'NO CONTENT' else row['Outside PDF Text'], axis=1)

# END OF STEP 6

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 6: CLEAN UP THE DATA

# Clean the text data in 'Inside PDF Text' and 'Text' and 'Outside PDF Text' columns by replacing newline characters
article_link_directory['Inside PDF Text'] = article_link_directory['Inside PDF Text'].str.replace('\n', ' ')
article_link_directory['Text'] = article_link_directory['Text'].str.replace('\n', ' ')
article_link_directory['Outside PDF Text'] = article_link_directory['Outside PDF Text'].str.replace('\n', ' ')

# Extract document date from article_link_directory['Publishing Dates']
# List to store extracted dates
Date = []

# Iterate through the 'Publishing Dates' column and extract dates based on the condition
# "Updated" dates are disregarded because they only seem to occur with certain documents of lesser importance and don't override the pdf included in content...so the pdf data reflects the original date

for date_string in article_link_directory['Publishing Dates']:
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

# Extract document types from article_link_directory['Publishing Dates']
# List to store extracted document types
Document_Type = []

# Iterate through the 'Publishing Dates' column and extract document types based on the condition
for date_string in article_link_directory['Publishing Dates']:
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
article_link_directory['Document_Type'] = Document_Type

# Extract document source from article_link_directory['Publishing Dates']
# List to store extracted sources
Source = []

# Iterate through the 'Publishing Dates' column and extract sources based on the condition
for date_string in article_link_directory['Publishing Dates']:
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

# Create a final dataframe containing the main columns from article_link_directory
swe_gov_docs = article_link_directory[['Source', 'Date', 'Document_Type', 'Text', 'Content Links']].copy()
swe_gov_docs.rename(columns={'Content Links': 'URL'}, inplace=True)

# END OF STEP 6

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 7: EXPORT DATA

# Export Data
swe_gov_docs.to_csv('scraped_publication_data')
swe_gov_docs.to_json('scraped_publication_data')