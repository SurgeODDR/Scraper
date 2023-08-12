import requests
import pandas as pd
import openai
from flask import Flask, jsonify
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from ratelimiter import RateLimiter
import io
import logging

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

credential = DefaultAzureCredential()
key_vault_uri = "https://Keyvaultxscrapingoddr.vault.azure.net/"
secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)
openai.api_key = secret_client.get_secret("openai-api-key").value

# Rate limiter for OpenAI API
rate_limiter = RateLimiter(max_calls=3500, period=60)

def openai_request(data):
    headers = {"Authorization": f"Bearer {openai.api_key}"}
    for _ in range(3):  # 3 retries
        with rate_limiter:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
            if response.status_code == 429:  # Rate Limit Exceeded
                app.logger.warning("Rate limit exceeded. Retrying...")
                continue
            response.raise_for_status()
            return response.json()
    app.logger.error("Failed to get a successful response from OpenAI.")
    return {}

def save_to_blob(blob_service_client, content, file_name):
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", file_name)
    blob_client.upload_blob(content, overwrite=True)

def get_processed_tweet_ids(blob_service_client):
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "processed_tweet_ids.txt")
    if blob_client.exists():
        processed_ids = set(map(int, blob_client.download_blob().readall().decode('utf-8').splitlines()))
        return processed_ids
    return set()

def analyze_text(text):
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [
            {
                "role": "system",
                "content": """
                    Extract mentions of politicians and celebrities from the text. For each mentioned figure, categorize sentiments, emotions, and specific keywords related to them. Present the results in CSV format:
                    
                    "Name, Sentiment/Emotion/Keyword, Count"
                    For example: 
                    "John Doe, Positive Sentiment, 5"
                    "Jane Smith, Happiness, 3"
                """
            },
            {"role": "user", "content": text}
        ],
        "temperature": 0.3,
        "max_tokens": 13000
    }

    response_data = openai_request(data)  # Using the function without unnecessary arguments
    if 'choices' in response_data:
        return response_data['choices'][0]['message']['content'].strip()
    return "Error analyzing the text."

import re

def flexible_column_matching(df, keyword):
    """Find the column name that best matches the keyword."""
    for col in df.columns:
        if keyword.lower() in col.lower():
            return col
    return None

def combine_and_save_analysis(blob_service_client, new_analysis):
    try:
        new_df = pd.read_csv(io.StringIO(new_analysis), index_col=0)

        # Standardize the index labels (remove counts)
        new_df.index = new_df.index.astype(str).str.split(',').str[0]

        celeb_db_analysis_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "celeb_db_analysis.csv")
        if celeb_db_analysis_blob_client.exists():
            existing_content = celeb_db_analysis_blob_client.download_blob().readall().decode('utf-8')
            existing_df = pd.read_csv(io.StringIO(existing_content), index_col=0)

            # Standardize the index labels of the existing df
            existing_df.index = existing_df.index.astype(str).str.split(',').str[0]

            combined_df = new_df.add(existing_df, fill_value=0)
        else:
            combined_df = new_df

        combined_csv_content = combined_df.to_csv()
        save_to_blob(blob_service_client, combined_csv_content, "celeb_db_analysis.csv")

    except Exception as e:
        app.logger.error(f"Error in combine_and_save_analysis: {str(e)}")

@app.route('/process', methods=['GET'])
def process_data():
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    
    # Fetch the tweets from Azure Blob Storage
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
    download_stream = blob_client.download_blob()
    data = download_stream.readall()
    df = pd.read_json(io.BytesIO(data))
    
    # Retrieve the list of tweet IDs that have been previously processed
    processed_ids = get_processed_tweet_ids(blob_service_client)

    # Filter out tweets that have been processed before
    df = df[~df['id'].isin(processed_ids)]
    
    # Split the remaining tweets into chunks for processing
    chunk_size = 5
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Process each chunk and update the aggregate analysis CSV file
    for chunk_df in chunks:
        chunk_text = chunk_df['text'].astype(str).str.cat(sep='\n')
        new_analysis = analyze_text(chunk_text)
        combine_and_save_analysis(blob_service_client, new_analysis)

        # Update the list of processed tweet IDs and save back to Azure Blob Storage
        processed_ids.update(chunk_df['id'].tolist())
        save_to_blob(blob_service_client, "\n".join(map(str, processed_ids)), "processed_tweet_ids.txt")

    return jsonify({'message': 'Data processed successfully'}), 200
