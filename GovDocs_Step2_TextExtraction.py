# GovDocsOperational.py
# 10/10/23

# DESCRIPTION
# Process the data scraped by GovDocs_Step1_Selenium

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
csv_file_path = "ppublication_info_1_9000.csv"  # Update with the correct file path
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
    time.sleep(randint(1,2))  # Adjust the delay time as needed

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