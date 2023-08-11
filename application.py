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

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

credential = DefaultAzureCredential()
key_vault_uri = "https://Keyvaultxscrapingoddr.vault.azure.net/"
secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)
openai.api_key = secret_client.get_secret("openai-api-key").value

# Rate limiter for the first OpenAI API key
rate_limiter = RateLimiter(max_calls=3500, period=60)

import time

def openai_request(data, api_key, rate_limiter_obj, retries=3, delay=5):
    headers = {"Authorization": f"Bearer {api_key}"}
    
    for _ in range(retries):
        with rate_limiter_obj:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
            if response.status_code == 429:  # Rate Limit Exceeded
                app.logger.warning("Rate limit exceeded. Retrying in {} seconds...".format(delay))
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response.json()
    app.logger.error("Failed to make a successful request after {} retries.".format(retries))
    return {}  # Return an empty dictionary to indicate failure


def save_to_blob(blob_service_client, content, file_name):
    path = f"/tmp/{file_name}"
    with open(path, 'w') as file:
        file.write(content)
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", file_name)
    with open(path, 'rb') as data:
        blob_client.upload_blob(data, overwrite=True)

def get_processed_tweet_ids(blob_service_client):
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "processed_tweet_ids.txt")
    if blob_client.exists():
        download_stream = blob_client.download_blob()
        processed_ids = set(map(int, download_stream.readall().decode('utf-8').splitlines()))
        return processed_ids
    return set()

def update_processed_tweet_ids(blob_service_client, processed_ids):
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "processed_tweet_ids.txt")
    ids_str = "\n".join(map(str, processed_ids))
    blob_client.upload_blob(ids_str, overwrite=True)

def analyze_text(text):
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [
            {"role": "system", "content": """
        Generate a quantitative analysis in CSV format based on the provided text. Cover:
        - Sentiments (Positive, Negative, Neutral)
        - Key emotions (happiness, sadness, anger, fear, surprise, disgust, jealousy, outrage/indignation, distrust/skepticism, despair/hopelessness, shock/astonishment, relief, and empowerment)
        - Keywords related to inequality, unfairness, distrust in government, unjust actions, disloyalty, and perceptions of corruption.
        
        CSV Structure:
        "Category, Total Mentions"
        "Sentiments: Positive, [Total Positive Sentiment Mentions]"
        "Sentiments: Negative, [Total Negative Sentiment Mentions]"
        "Sentiments: Neutral, [Total Neutral Sentiment Mentions]"
        "Emotions: Happiness, [Total Happiness Mentions]"
        ...
        "Keywords: Inequality, [Total Inequality Mentions]"
        ...
        """
            },
            {"role": "user", "content": text}
        ],
        "temperature": 0.3,
        "max_tokens": 12000
    }
    
    response_data = openai_request(data, openai.api_key, rate_limiter)
    return response_data['choices'][0]['message']['content'].strip() if 'choices' in response_data else "Error analyzing the text."

def combine_and_save_analysis(blob_service_client, new_analysis):
    # Save the new analysis as new_analysis.csv
    save_to_blob(blob_service_client, new_analysis, "new_analysis.csv")

    # Load new_analysis.csv into a dataframe
    new_df = pd.read_csv(io.StringIO(new_analysis), index_col=0)

    # Fetch and load db_analysis.csv into a dataframe, if it exists
    db_analysis_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "db_analysis.csv")
    if db_analysis_blob_client.exists():
        existing_content = db_analysis_blob_client.download_blob().readall().decode('utf-8')
        existing_df = pd.read_csv(io.StringIO(existing_content), index_col=0)
        # Sum values based on the category
        combined_df = new_df.add(existing_df, fill_value=0)
    else:
        combined_df = new_df

    # Save the combined dataframe to db_analysis.csv
    combined_csv_content = combined_df.to_csv()
    save_to_blob(blob_service_client, combined_csv_content, "db_analysis.csv")

@app.route('/process', methods=['GET'])
def process_data():
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
    download_stream = blob_client.download_blob()
    data = download_stream.readall()
    df = pd.read_json(io.BytesIO(data))
    
    # Get the list of tweet IDs that have been processed already
    processed_ids = get_processed_tweet_ids(blob_service_client)

    # Filter out tweets that have been processed
    df = df[~df['id'].isin(processed_ids)]
    
    text = df['text'].str.cat(sep='\n')
    analysis = analyze_text(text)

    # Combine the new analysis with existing data and save
    combine_and_save_analysis(blob_service_client, analysis)

    # Update the list of processed tweets
    processed_ids.update(df['id'].tolist())
    update_processed_tweet_ids(blob_service_client, processed_ids)

    return jsonify({'message': 'Data processed successfully'}), 200

if __name__ == "__main__":
    app.run(debug=True)
