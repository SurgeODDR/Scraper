import requests
from flask import Flask, jsonify
import pandas as pd
import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
import io
import logging
import time
from ratelimiter import RateLimiter

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

credential = DefaultAzureCredential()
key_vault_uri = "https://Keyvaultxscrapingoddr.vault.azure.net/"
secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)
openai.api_key = secret_client.get_secret("openai-api-key").value
rate_limiter = RateLimiter(max_calls=3500, period=60)

def openai_request(data, api_key, rate_limiter_obj):
    headers = {"Authorization": f"Bearer {api_key}"}
    with rate_limiter_obj:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()

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
    analysis_content = response_data['choices'][0]['message']['content'].strip() if 'choices' in response_data else "Error analyzing the text."
    
    df = pd.read_csv(io.StringIO(analysis_content), index_col=0)
    category_counts = df.to_dict()["Total Mentions"]
    
    return analysis_content, category_counts

def combine_and_save_analysis(blob_service_client, new_analysis):
    new_df = pd.read_csv(io.StringIO(new_analysis), index_col=0)
    db_analysis_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "db_analysis.csv")
    if db_analysis_blob_client.exists():
        existing_content = db_analysis_blob_client.download_blob().readall().decode('utf-8')
        existing_df = pd.read_csv(io.StringIO(existing_content), index_col=0)
        combined_df = new_df.add(existing_df, fill_value=0)
    else:
        combined_df = new_df
    combined_csv_content = combined_df.to_csv()
    save_to_blob(blob_service_client, combined_csv_content, "db_analysis.csv")

def update_audit_log(blob_service_client, num_tweets, start_id, end_id, mentions_details):
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    new_log = f"{current_time},{num_tweets},{start_id},{end_id},{mentions_details}\n"
    
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "audit.csv")
    if blob_client.exists():
        app.logger.info("audit.csv exists. Fetching its content.")
        current_content = blob_client.download_blob().readall().decode('utf-8')
        updated_content = current_content + new_log
    else:
        app.logger.info("audit.csv doesn't exist. Creating a new one.")
        header = "Timestamp,NumOfTweets,StartTweetID,EndTweetID,MentionsDetails\n"
        updated_content = header + new_log
        blob_client.upload_blob(header)  # Explicitly creating the file with the header.
        time.sleep(2)  # Small delay to ensure the file is created before appending.

    app.logger.info("Saving updated content to audit.csv.")
    save_to_blob(blob_service_client, updated_content, "audit.csv")
    time.sleep(2)  # Small delay to ensure no overlapping writes.

@app.route('/process', methods=['GET'])
def process_data():
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
    download_stream = blob_client.download_blob()
    data = download_stream.readall()
    df = pd.read_json(io.BytesIO(data))
    
    processed_ids = get_processed_tweet_ids(blob_service_client)
    df = df[~df['id'].isin(processed_ids)]
    
    chunk_size = 5
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    for chunk_df in chunks:
        chunk_text = chunk_df['text'].str.cat(sep='\n')
        new_analysis, category_counts = analyze_text(chunk_text)
        combine_and_save_analysis(blob_service_client, new_analysis)
        
        processed_ids.update(chunk_df['id'].tolist())
        update_processed_tweet_ids(blob_service_client, processed_ids)
        
        mentions_details = "; ".join([f"{key}: {value}" for key, value in category_counts.items()])
        start_id = chunk_df['id'].iloc[0]
        end_id = chunk_df['id'].iloc[-1]
        update_audit_log(blob_service_client, len(chunk_df), start_id, end_id, mentions_details)

    return jsonify({'message': 'Data processed successfully'}), 200

if __name__ == "__main__":
    app.run(debug=True)
