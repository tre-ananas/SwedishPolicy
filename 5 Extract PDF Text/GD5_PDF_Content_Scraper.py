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

# END OF STEP 1

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 2: LOAD DATA

# Load data from CSV file into a DataFrame
article_link_directory = pd.read_csv('article_link_directory.csv')

# END OF STEP 2

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 3: COLLECT PDF CONTENT

# Specify the path to the CSV file
csv_file_path = "pdf_data.csv"

# Open the CSV file in write mode
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    # Create a CSV writer object
    csv_writer = csv.writer(csv_file)

    # Write the header row to the CSV file
    csv_writer.writerow(['PDF URL', 'Inside PDF Text'])

    for i, pdf_url in enumerate(tqdm(article_link_directory['Full Collected Links'], desc="Step 2: Collecting PDF Text", unit="link")):
        # Check if the URL contains the NO CONTENT alert
        if "NO CONTENT" in pdf_url:
            # Write the data to the CSV file
            csv_writer.writerow([pdf_url, "NO CONTENT"])
            continue

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

                    # Write the data to the CSV file
                    csv_writer.writerow([pdf_url, text_data])
                else:
                    print(f"Failed to fetch PDF URL: {pdf_url}, Status code: {response.status_code}")
                    # Write the data to the CSV file
                    csv_writer.writerow([pdf_url, "NO CONTENT"])

            except Exception as e:
                print(f"An error occurred while processing PDF URL: {pdf_url}, Error: {str(e)}")
                # Write the data to the CSV file
                csv_writer.writerow([pdf_url, "NO CONTENT"])

        # Introduce a delay time before the next request
        time.sleep(3)  # Adjust the delay time as needed

        # Add a break statement if the loop index is equal to the expected number of links minus 1
        if i == len(article_link_directory['Full Collected Links']) - 1:
            break


# END OF STEP 3

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------