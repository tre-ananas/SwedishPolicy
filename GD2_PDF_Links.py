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
import csv

# Other Settings
# pd.set_option('display.max.colwidth', None) # max display width

# END OF STEP 1

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 2: LOAD DATA

# Load data from CSV file into a DataFrame
csv_file_path = "publication_info.csv"
existing_data = pd.read_csv(csv_file_path)

# Rename the columns to match your desired names
existing_data.columns = ['Content Links', 'Publishing Dates']

# Append the loaded data to the existing DataFrame
article_link_directory = existing_data.copy()

# END OF STEP 2

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 3: COLLECT THE PDF LINKS FROM EACH CONTENT PAGE CONTAINING A PDF; IF A CONTENT PAGE CONTAINS NO PDF, GRAB ALL TEXT INSTEAD

# No content alert
no_content_alert = "NO CONTENT"

# Add new columns to the DataFrame
article_link_directory['Collected Links'] = ""
article_link_directory['Outside PDF Text'] = ""

# Specify the CSV file path
csv_file_path = "pdf_links.csv"



# Open the CSV file in write mode with a CSV writer
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write the header row to the CSV file
    csv_writer.writerow(["Collected Links", "Outside PDF Text"])

    for i, link in enumerate(tqdm(article_link_directory['Content Links'], desc="Step 1: Collecting PDF Links", unit="link")):
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
                    if first_link:
                        article_link_directory.at[i, 'Collected Links'] = first_link['href']
                        article_link_directory.at[i, 'Outside PDF Text'] = no_content_alert
                    else:
                        article_link_directory.at[i, 'Collected Links'] = no_content_alert
                else:
                    # If <ul> tag is not found, extract text from <p> tags
                    paragraphs = soup.find_all('p')
                    text_data = '\n'.join([p.get_text(strip=True) for p in paragraphs])
                    article_link_directory.at[i, 'Outside PDF Text'] = text_data
                    article_link_directory.at[i, 'Collected Links'] = no_content_alert
            else:
                print(f"Failed to fetch URL: {link}, Status code: {response.status_code}")
                # If <ul> tag is not found, extract text from <p> tags
                paragraphs = soup.find_all('p')
                text_data = '\n'.join([p.get_text(strip=True) for p in paragraphs])
                article_link_directory.at[i, 'Outside PDF Text'] = text_data
                article_link_directory.at[i, 'Collected Links'] = no_content_alert

            # Write the data to the CSV file in each iteration
            csv_writer.writerow([article_link_directory.at[i, 'Collected Links'],
                                 article_link_directory.at[i, 'Outside PDF Text']])
            
            # Print the data for the current iteration
            print(f"Collected Links: {article_link_directory.at[i, 'Collected Links']}")
            print(f"Outside PDF Text: {article_link_directory.at[i, 'Outside PDF Text']}")
            print("----------------------------------------")

        except Exception as e:
            print(f"An error occurred while processing URL: {link}, Error: {str(e)}")
            # If an error occurs, write the available data to the CSV file before continuing
            csv_writer.writerow([no_content_alert, no_content_alert])

        # Introduce a random delay time before the next request
        time.sleep(5)

        # Add a break statement if the loop index is equal to the expected number of links minus 1
        if i == len(article_link_directory['Content Links']) - 1:
            break

# END OF STEP 3

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------