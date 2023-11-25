# SentimentAnalysis.py
# 11/20/23

# DESCRIPTION
# This code takes .csv data produced by the GovDocsOperational.py script and outputs a .csv with sentiment and keyword data.
# This data can be loaded into R or Python for further analysis, including keyword frequency/proportion, sentiment by keyword presence, and sentiment across time.

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 1: LIBRARIES AND SETTINGS

# Import Packages and Model Commands
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch
import pandas as pd
import nltk
from tqdm import tqdm
import datetime
import sys
from nltk.corpus import stopwords

# Swedish stopwords
nltk.download('stopwords')
stop_words_swedish = set(stopwords.words('swedish'))

# Download the NLTK sentence tokenizer data
nltk.download("punkt")

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 2: LOAD MODELS FROM HUGGING FACE - RECORDED FUTURE

# Load pre-trained model and tokenizer from Recorded Future
tokenizer = BertTokenizerFast.from_pretrained("RecordedFuture/Swedish-Sentiment-Fear")
classifier_fear= BertForSequenceClassification.from_pretrained("RecordedFuture/Swedish-Sentiment-Fear")
classifier_violence = BertForSequenceClassification.from_pretrained("RecordedFuture/Swedish-Sentiment-Violence")

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 3: LOAD DATA

# Check if the correct number of command-line arguments are provided
if len(sys.argv) != 3:
    print("Usage: python script_name.py input_file_name.csv output_file_name.csv")
    sys.exit(1)

# Get input and output file names from command-line arguments
input_file_name = sys.argv[1]
output_file_name = sys.argv[2]

# Load Data
try:
    unprocessed_data = pd.read_csv(input_file_name)
    unprocessed_data = unprocessed_data.rename(columns={"Unnamed: 0": "Index"})
    unprocessed_data = unprocessed_data[unprocessed_data['Text'] != 'NO CONTENT']
except FileNotFoundError:
    print(f"Error: Input file '{input_file_name}' not found.")
    sys.exit(1)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 4: SENTENCE CLASSIFICATION

# Function to handle errors during processing
def process_text_with_fail_safes(text, text_id):
    try:
        sentences = nltk.sent_tokenize(text)

        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)

            # Calculate probabilities for classifier_fear
            outputs_fear = classifier_fear(**inputs)
            probabilities_fear = torch.nn.functional.softmax(outputs_fear.logits, dim=1).tolist()[0]

            # Calculate probabilities for classifier_violence
            outputs_violence = classifier_violence(**inputs)
            probabilities_violence = torch.nn.functional.softmax(outputs_violence.logits, dim=1).tolist()[0]

            texts_list.append(sentence)
            ids_list.append(text_id)

            probabilities_fear_list.append(probabilities_fear)
            probabilities_violence_list.append(probabilities_violence)

    except Exception as e:
        print(f"Error processing text with ID {text_id}: {e}")
        # If there's an error, replace probabilities with 0
        probabilities_fear_list.append([0, 0, 0])
        probabilities_violence_list.append([0, 0, 0])
        texts_list.append("Error processing text")
        ids_list.append(text_id)

# Lists to store probabilities, texts, and IDs for both classifiers
probabilities_fear_list = []
probabilities_violence_list = []
texts_list = []
ids_list = []

for text, text_id in tqdm(zip(unprocessed_data['Text'].tolist(), unprocessed_data['Index'].tolist()),
                          total=len(unprocessed_data), desc="Processing Texts"):
    process_text_with_fail_safes(text, text_id)

# Creating the classified_sentence_data DataFrame
classified_sentence_data = pd.DataFrame({
    'Fear Class 0 Probability': [item[0] for item in probabilities_fear_list],
    'Fear Class 1 Probability': [item[1] for item in probabilities_fear_list],
    'Fear Class 2 Probability': [item[2] for item in probabilities_fear_list],
    'Violence Class 0 Probability': [item[0] for item in probabilities_violence_list],
    'Violence Class 1 Probability': [item[1] for item in probabilities_violence_list],
    'Violence Class 2 Probability': [item[2] for item in probabilities_violence_list],
    'Text': texts_list,
    'ID': ids_list
})

# Merging additional columns from unprocessed_data based on 'Index' and 'ID'
classified_sentence_data = pd.merge(classified_sentence_data, unprocessed_data[['Index', 'Date', 'Document Type', 'Source', 'PDF Indicator', 'Content Page', 'PDF Link']], 
                                    left_on='ID', right_on='Index', how='left')

# Dropping the redundant 'Index' column
classified_sentence_data.drop(columns=['Index'], inplace=True)

# Reordering the columns
classified_sentence_data = classified_sentence_data[['ID', 'Date',
                                                     'Fear Class 0 Probability', 'Fear Class 1 Probability', 'Fear Class 2 Probability',
                                                     'Violence Class 0 Probability', 'Violence Class 1 Probability', 'Violence Class 2 Probability',
                                                     'Document Type', 'Source', 'PDF Indicator', 'Text', 'Content Page', 'PDF Link']]

# STEP 5: CLASSIFY ARTICLES BY TOP SENTENCES

# Group result_df by 'ID'
classified_sentence_data_grouped = classified_sentence_data.groupby('ID')

# Lists to store data for the new DataFrame for fear probabilities
new_ids = []
new_texts = []
new_fear_class_0_probs = []
new_fear_class_1_probs = []
new_fear_class_2_probs = []
new_violence_class_0_probs = []
new_violence_class_1_probs = []
new_violence_class_2_probs = []

# Fear Classification
# Iterate through grouped_df
for id, group in tqdm(classified_sentence_data_grouped, desc="Processing Groups"):
    num_rows = len(group)
    # If there are less than 10 rows, select the row with the lowest Class 0 probability
    if num_rows < 10:
        min_index = group['Fear Class 0 Probability'].idxmin()
        selected_row = group.loc[min_index]
        new_ids.append(id)
        new_texts.append(selected_row['Text'])
        new_fear_class_0_probs.append(selected_row['Fear Class 0 Probability'])
        new_fear_class_1_probs.append(selected_row['Fear Class 1 Probability'])
        new_fear_class_2_probs.append(selected_row['Fear Class 2 Probability'])
    # If there are 10-49 rows, select the two rows with the lowest Class 0 probabilities
    elif 10 <= num_rows < 50:
        selected_rows = group.nsmallest(2, 'Fear Class 0 Probability')
        avg_class_0_prob = selected_rows['Fear Class 0 Probability'].mean()
        avg_class_1_prob = selected_rows['Fear Class 1 Probability'].mean()
        avg_class_2_prob = selected_rows['Fear Class 2 Probability'].mean()
        new_ids.append(id)
        new_texts.append(selected_rows['Text'].values[0])  # Select the text from the first row
        new_fear_class_0_probs.append(avg_class_0_prob)
        new_fear_class_1_probs.append(avg_class_1_prob)
        new_fear_class_2_probs.append(avg_class_2_prob)
    # If there are 50-99 rows, select the three rows with the lowest Class 0 probabilities
    elif 50 <= num_rows < 100:
        selected_rows = group.nsmallest(3, 'Fear Class 0 Probability')
        avg_class_0_prob = selected_rows['Fear Class 0 Probability'].mean()
        avg_class_1_prob = selected_rows['Fear Class 1 Probability'].mean()
        avg_class_2_prob = selected_rows['Fear Class 2 Probability'].mean()
        new_ids.append(id)
        new_texts.append(selected_rows['Text'].values[0])  # Select the text from the first row
        new_fear_class_0_probs.append(avg_class_0_prob)
        new_fear_class_1_probs.append(avg_class_1_prob)
        new_fear_class_2_probs.append(avg_class_2_prob)
    # If there are 100-199 rows, select the four rows with the lowest Class 0 probabilities
    elif 100 <= num_rows < 200:
        selected_rows = group.nsmallest(4, 'Fear Class 0 Probability')
        avg_class_0_prob = selected_rows['Fear Class 0 Probability'].mean()
        avg_class_1_prob = selected_rows['Fear Class 1 Probability'].mean()
        avg_class_2_prob = selected_rows['Fear Class 2 Probability'].mean()
        new_ids.append(id)
        new_texts.append(selected_rows['Text'].values[0])  # Select the text from the first row
        new_fear_class_0_probs.append(avg_class_0_prob)
        new_fear_class_1_probs.append(avg_class_1_prob)
        new_fear_class_2_probs.append(avg_class_2_prob)
    # If there are 200 or more rows, select the five rows with the lowest Class 0 probabilities
    else:
        selected_rows = group.nsmallest(5, 'Fear Class 0 Probability')
        avg_class_0_prob = selected_rows['Fear Class 0 Probability'].mean()
        avg_class_1_prob = selected_rows['Fear Class 1 Probability'].mean()
        avg_class_2_prob = selected_rows['Fear Class 2 Probability'].mean()
        new_ids.append(id)
        new_texts.append(selected_rows['Text'].values[0])  # Select the text from the first row
        new_fear_class_0_probs.append(avg_class_0_prob)
        new_fear_class_1_probs.append(avg_class_1_prob)
        new_fear_class_2_probs.append(avg_class_2_prob)

# Lists to store data for the new DataFrame for violence probabilities
new_ids = []
new_texts = []
new_violence_class_0_probs = []
new_violence_class_1_probs = []
new_violence_class_2_probs = []

# Violence Classification
# Iterate through grouped_df
for id, group in tqdm(classified_sentence_data_grouped, desc="Processing Groups"):
    num_rows = len(group)
    # If there are less than 10 rows, select the row with the lowest Class 0 probability
    if num_rows < 10:
        min_index = group['Violence Class 0 Probability'].idxmin()
        selected_row = group.loc[min_index]
        new_ids.append(id)
        new_texts.append(selected_row['Text'])
        new_violence_class_0_probs.append(selected_row['Violence Class 0 Probability'])
        new_violence_class_1_probs.append(selected_row['Violence Class 1 Probability'])
        new_violence_class_2_probs.append(selected_row['Violence Class 2 Probability'])
    # If there are 10-49 rows, select the two rows with the lowest Class 0 probabilities
    elif 10 <= num_rows < 50:
        selected_rows = group.nsmallest(2, 'Violence Class 0 Probability')
        avg_class_0_prob = selected_rows['Violence Class 0 Probability'].mean()
        avg_class_1_prob = selected_rows['Violence Class 1 Probability'].mean()
        avg_class_2_prob = selected_rows['Violence Class 2 Probability'].mean()
        new_ids.append(id)
        new_texts.append(selected_rows['Text'].values[0])  # Select the text from the first row
        new_violence_class_0_probs.append(avg_class_0_prob)
        new_violence_class_1_probs.append(avg_class_1_prob)
        new_violence_class_2_probs.append(avg_class_2_prob)
    # If there are 50-99 rows, select the three rows with the lowest Class 0 probabilities
    elif 50 <= num_rows < 100:
        selected_rows = group.nsmallest(3, 'Violence Class 0 Probability')
        avg_class_0_prob = selected_rows['Violence Class 0 Probability'].mean()
        avg_class_1_prob = selected_rows['Violence Class 1 Probability'].mean()
        avg_class_2_prob = selected_rows['Violence Class 2 Probability'].mean()
        new_ids.append(id)
        new_texts.append(selected_rows['Text'].values[0])  # Select the text from the first row
        new_violence_class_0_probs.append(avg_class_0_prob)
        new_violence_class_1_probs.append(avg_class_1_prob)
        new_violence_class_2_probs.append(avg_class_2_prob)
    # If there are 100-199 rows, select the four rows with the lowest Class 0 probabilities
    elif 100 <= num_rows < 200:
        selected_rows = group.nsmallest(4, 'Violence Class 0 Probability')
        avg_class_0_prob = selected_rows['Violence Class 0 Probability'].mean()
        avg_class_1_prob = selected_rows['Violence Class 1 Probability'].mean()
        avg_class_2_prob = selected_rows['Violence Class 2 Probability'].mean()
        new_ids.append(id)
        new_texts.append(selected_rows['Text'].values[0])  # Select the text from the first row
        new_violence_class_0_probs.append(avg_class_0_prob)
        new_violence_class_1_probs.append(avg_class_1_prob)
        new_violence_class_2_probs.append(avg_class_2_prob)
    # If there are 200 or more rows, select the five rows with the lowest Class 0 probabilities
    else:
        selected_rows = group.nsmallest(5, 'Violence Class 0 Probability')
        avg_class_0_prob = selected_rows['Violence Class 0 Probability'].mean()
        avg_class_1_prob = selected_rows['Violence Class 1 Probability'].mean()
        avg_class_2_prob = selected_rows['Violence Class 2 Probability'].mean()
        new_ids.append(id)
        new_texts.append(selected_rows['Text'].values[0])  # Select the text from the first row
        new_violence_class_0_probs.append(avg_class_0_prob)
        new_violence_class_1_probs.append(avg_class_1_prob)
        new_violence_class_2_probs.append(avg_class_2_prob)

# Create a new DataFrame containing the original unprocessed data but with the representative classification scores
processed_data = pd.DataFrame({
    'ID': new_ids,
    'Date': unprocessed_data['Date'],
    'Fear Class 0 Probability': new_fear_class_0_probs,
    'Fear Class 1 Probability': new_fear_class_1_probs,
    'Fear Class 2 Probability': new_fear_class_2_probs,
    'Violence Class 0 Probability': new_violence_class_0_probs,
    'Violence Class 1 Probability': new_violence_class_1_probs,
    'Violence Class 2 Probability': new_violence_class_2_probs,
    'Document Type': unprocessed_data['Document Type'],
    'Source': unprocessed_data['Source'],
    'Text': unprocessed_data['Text'],
    'PDF Indicator': unprocessed_data['PDF Indicator'],
    'Content Page': unprocessed_data['Content Page'],
    'PDF Link': unprocessed_data['PDF Link']
})

# STEP 6: CLASSIFY ARTICLES AS A WHOLE WITH NO STOPWORDS

# Lists to store probabilities, texts, and IDs for both classifiers
probabilities_fear_list_2 = []
probabilities_violence_list_2 = []
texts_list_2 = []
ids_list_2 = []

text_entries_2 = unprocessed_data['Text'].tolist()
ids_2 = unprocessed_data['Index'].tolist()

for text, text_id in tqdm(zip(text_entries_2, ids_2), total=len(text_entries_2), desc="Processing Texts"):
    try:
        # Lowercase the entire text
        text_lower = text.lower()

        # Tokenize and remove stop words
        tokens = nltk.word_tokenize(text_lower)
        tokens_filtered = [word for word in tokens if word.isalnum() and word not in stop_words_swedish]

        # Reconstruct the text
        text_filtered = ' '.join(tokens_filtered)

        # Process the entire text at once
        inputs = tokenizer(text_filtered, return_tensors="pt", truncation=True, max_length=512)

        # Calculate probabilities for classifier_fear
        outputs_fear = classifier_fear(**inputs)
        probabilities_fear = torch.nn.functional.softmax(outputs_fear.logits, dim=1).tolist()[0]
        probabilities_fear_list_2.append(probabilities_fear)

        # Calculate probabilities for classifier_violence
        outputs_violence = classifier_violence(**inputs)
        probabilities_violence = torch.nn.functional.softmax(outputs_violence.logits, dim=1).tolist()[0]
        probabilities_violence_list_2.append(probabilities_violence)

        texts_list_2.append(text_filtered)
        ids_list_2.append(text_id)
    except Exception as e:
        # If an error occurs, print the error and append zeros to the probability lists
        print(f"Error processing text with ID {text_id}: {e}")
        probabilities_fear_list_2.append([0.0, 0.0, 0.0])
        probabilities_violence_list_2.append([0.0, 0.0, 0.0])
        texts_list_2.append("Error processing text")
        ids_list_2.append(text_id)

# Now, probabilities_fear_list_2 and probabilities_violence_list_2 contain probabilities
# for classifier_fear and classifier_violence respectively, for each text.

# Creating the classified_sentence_data DataFrame
classified_sentence_data_2 = pd.DataFrame({
    'Fear Class 0 Probability_2': [item[0] for item in probabilities_fear_list_2],
    'Fear Class 1 Probability_2': [item[1] for item in probabilities_fear_list_2],
    'Fear Class 2 Probability_2': [item[2] for item in probabilities_fear_list_2],
    'Violence Class 0 Probability_2': [item[0] for item in probabilities_violence_list_2],
    'Violence Class 1 Probability_2': [item[1] for item in probabilities_violence_list_2],
    'Violence Class 2 Probability_2': [item[2] for item in probabilities_violence_list_2],
    'Text_2': texts_list_2,
    'ID_2': ids_list_2
})

# STEP 7: MERGE DATA

# Columns to be joined from classified_sentence_data_2
columns_to_join = ['ID_2', 'Fear Class 0 Probability_2', 'Fear Class 1 Probability_2', 'Fear Class 2 Probability_2',
                    'Violence Class 0 Probability_2', 'Violence Class 1 Probability_2', 'Violence Class 2 Probability_2']

# Merge processed_data with the selected columns from classified_sentence_data_2
fully_classified_data = pd.merge(processed_data, classified_sentence_data_2[columns_to_join], how='left', left_on='ID', right_on='ID_2')

# Drop the redundant 'ID_2' column
fully_classified_data.drop(columns=['ID_2'], inplace=True)

# STEP 13: SAVE DATA TO .CSV

# Save processed data to CSV
fully_classified_data.to_csv(output_file_name, index=True)

print(f"Processed data has been saved as {output_file_name}")