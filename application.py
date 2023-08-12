import requests
from flask import Flask, jsonify
import pandas as pd
import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
import io
import logging
from ratelimiter import RateLimiter

# Initialization
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

credential = DefaultAzureCredential()
key_vault_uri = "https://Keyvaultxscrapingoddr.vault.azure.net/"
secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)
openai.api_key = secret_client.get_secret("openai-api-key").value

rate_limiter = RateLimiter(max_calls=3500, period=60)

# Functions
def openai_request(data):
    headers = {"Authorization": f"Bearer {openai.api_key}"}
    url = "https://api.openai.com/v1/chat/completions"
    
    with rate_limiter:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get('choices', [{}])[0].get('message', {}).get('content', "").strip()

def save_to_blob(blob_service_client, content, file_name):
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", file_name)
    blob_client.upload_blob(content, overwrite=True)

def get_processed_tweet_ids(blob_service_client):
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "processed_tweet_ids.txt")
    if blob_client.exists():
        ids = blob_client.download_blob().readall().decode('utf-8').splitlines()
        return set(map(int, ids))
    return set()

def update_processed_tweet_ids(blob_service_client, processed_ids):
    ids_str = "\n".join(map(str, processed_ids))
    save_to_blob(blob_service_client, ids_str, "processed_tweet_ids.txt")

def analyze_text(text):
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [
            {"role": "system", "content": """
                    Generate a quantitative analysis in CSV format based on the provided text. Cover:
                    - Sentiments regarding politicians and celebrities (Positive, Negative, Neutral)
                    - Key emotions regarding politicians and celebrities (happiness, sadness, anger, fear, surprise, disgust, jealousy, outrage/indignation, distrust/skepticism, despair/hopelessness, shock/astonishment, relief, and empowerment)
                    - Keywords regarding politicians and celebrities related to inequality, unfairness, distrust in government, unjust actions, disloyalty, and perceptions of corruption.
                    
                    CSV Structure:
                    "Name, Category, Total Mentions"
                    "[Politician/Celebrity Name], Sentiments: Positive, [Total Positive Sentiment Mentions for this individual]"
                    "[Politician/Celebrity Name], Sentiments: Negative, [Total Negative Sentiment Mentions for this individual]"
                    "[Politician/Celebrity Name], Sentiments: Neutral, [Total Neutral Sentiment Mentions for this individual]"
                    "[Politician/Celebrity Name], Emotions: Happiness, [Total Happiness Mentions for this individual]"
                    "[Politician/Celebrity Name], Keywords: Inequality, [Total Inequality Mentions for this individual]"
            """ },
            {"role": "user", "content": text}
        ],
        "temperature": 0.1,
        "max_tokens": 13000
    }
    return openai_request(data)

def clean_and_format_data(df):
    df.index = df.index.str.lower()
    df_grouped = df.groupby(df.index).sum()
    df_cleaned = df_grouped.reset_index()

    if len(df_cleaned.columns) == 2:
        df_cleaned.columns = ['Name or Category', 'Total Mentions']
    else:
        app.logger.warning("Unexpected number of columns in df_cleaned. Skipping column renaming.")

    return df_cleaned

def combine_and_save_analysis(blob_service_client, new_analysis):
    new_df = pd.read_csv(io.StringIO(new_analysis), index_col=0)
    new_df.index = new_df.index.str.split(',').str[0]

    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "db_analysis.csv")
    if blob_client.exists():
        existing_content = blob_client.download_blob().readall().decode('utf-8')
        existing_df = pd.read_csv(io.StringIO(existing_content), index_col=0)
        existing_df.index = existing_df.index.str.split(',').str[0]
        combined_df = new_df.add(existing_df, fill_value=0)
    else:
        combined_df = new_df

    cleaned_combined_df = clean_and_format_data(combined_df)
    save_to_blob(blob_service_client, cleaned_combined_df.to_csv(), "c_db_analysis.csv")

@app.route('/process', methods=['GET'])
def process_data():
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")

    if not blob_client.exists():
        return jsonify({'message': 'No data available'}), 404

    data = blob_client.download_blob().readall()
    df = pd.read_json(io.BytesIO(data))
    
    processed_ids = get_processed_tweet_ids(blob_service_client)
    df = df[~df['id'].isin(processed_ids)]
    
    chunk_size = 5
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    for chunk_df in chunks:
        chunk_text = chunk_df['text'].str.cat(sep='\n')
        new_analysis = analyze_text(chunk_text)
        combine_and_save_analysis(blob_service_client, new_analysis)
        processed_ids.update(chunk_df['id'].tolist())
        update_processed_tweet_ids(blob_service_client, processed_ids)

    return jsonify({'message': 'Data processed
