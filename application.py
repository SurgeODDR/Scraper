import json
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

def openai_request(data):
    headers = {"Authorization": f"Bearer {openai.api_key}"}
    url = "https://api.openai.com/v1/chat/completions"
    
    with rate_limiter:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get('choices', [{}])[0].get('message', {}).get('content', "").strip()

import logging

def save_to_blob(blob_service_client, content, file_name):
    try:
        blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", file_name)
        blob_client.upload_blob(content, overwrite=True)
        logging.info(f"Successfully saved content to blob with filename: {file_name}")
    except Exception as e:
        logging.error(f"Error saving content to blob: {e}")
        raise

def get_processed_tweet_ids(blob_service_client):
    try:
        blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "processed_tweet_ids.txt")
        if blob_client.exists():
            ids = blob_client.download_blob().readall().decode('utf-8').splitlines()
            return set(map(int, ids))
        return set()
    except Exception as e:
        logging.error(f"Error fetching processed tweet IDs: {e}")
        raise

def update_processed_tweet_ids(blob_service_client, processed_ids):
    try:
        ids_str = "\n".join(map(str, processed_ids))
        save_to_blob(blob_service_client, ids_str, "processed_tweet_ids.txt")
        logging.info("Successfully updated processed tweet IDs.")
    except Exception as e:
        logging.error(f"Error updating processed tweet IDs: {e}")
        raise

def combine_json_data(new_data, existing_data):
    try:
        for key, value in new_data.items():
            if key in existing_data:
                for sub_key, sub_value in value.items():
                    if sub_key in existing_data[key]:
                        for inner_key, inner_value in sub_value.items():
                            if inner_key in existing_data[key][sub_key]:
                                existing_data[key][sub_key][inner_key] += inner_value
                            else:
                                existing_data[key][sub_key][inner_key] = inner_value
                    else:
                        existing_data[key][sub_key] = sub_value
            else:
                existing_data[key] = value
        logging.info("Successfully combined JSON data.")
        return existing_data
    except Exception as e:
        logging.error(f"Error combining JSON data: {e}")
        raise


def analyze_text(text):
    # Define request payload
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [
            {"role": "system", "content": """
                    Generate a quantitative analysis in JSON format based on the provided text. Cover:
                    - Sentiments regarding politicians and celebrities (Positive, Negative, Neutral)
                    - Key emotions regarding politicians and celebrities (happiness, sadness, anger, fear, surprise, disgust, jealousy, outrage/indignation, distrust/skepticism, despair/hopelessness, shock/astonishment, relief, and empowerment)
                    - Keywords regarding politicians and celebrities related to inequality, unfairness, distrust in government, unjust actions, disloyalty, and perceptions of corruption.
                    
                    JSON Structure:
                    {
                        "[Politician/Celebrity Name]": {
                            "Sentiments": {
                                "Positive": [Total Positive Sentiment Mentions for this individual],
                                "Negative": [Total Negative Sentiment Mentions for this individual],
                                "Neutral": [Total Neutral Sentiment Mentions for this individual]
                            },
                            "Emotions": {
                                "Happiness": [Total Happiness Mentions for this individual],
                                ... [Other emotions]
                            },
                            "Keywords": {
                                "Inequality": [Total Inequality Mentions for this individual],
                                ... [Other keywords]
                            }
                        },
                        ... [Other Politician/Celebrity Names]
                    }
            """},
            {"role": "user", "content": text}
        ],
        "temperature": 0.1,
        "max_tokens": 13000
    }
    return openai_request(data)

import logging

def process_tweets_chunk(chunk, blob_service_client, processed_tweet_ids):
    combined_data_chunk = {}
    new_processed_ids = []
    
    # Initialize logger
    logger = logging.getLogger('process_tweets_chunk')
    logger.setLevel(logging.INFO)
    
    for _, row in chunk.iterrows():
        tweet_id = row['id']
        tweet_text = row['text']

        # Skip if the tweet is already processed
        if tweet_id in processed_tweet_ids:
            logger.info(f"Tweet with ID {tweet_id} already processed. Skipping.")
            continue

        try:
            analysis = json.loads(analyze_text(tweet_text))
            combined_data_chunk = combine_json_data(analysis, combined_data_chunk)
            
            # Add the tweet ID to the new_processed_ids list
            new_processed_ids.append(tweet_id)
            logger.info(f"Tweet with ID {tweet_id} processed successfully.")
        except Exception as e:
            logger.error(f"An error occurred while processing tweet with ID {tweet_id}: {e}")
            continue

    return combined_data_chunk, new_processed_ids

@app.route('/process', methods=['GET'])
def process_data():
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)

    try:
        blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
        download_stream = blob_client.download_blob()
        data = download_stream.readall()
        df = pd.read_json(io.BytesIO(data))

        processed_tweet_ids = get_processed_tweet_ids(blob_service_client)
        chunks = [df.iloc[i:i+3] for i in range(0, df.shape[0], 3)]

        for chunk in chunks:
            combined_data_chunk, new_processed_ids_chunk = process_tweets_chunk(chunk, blob_service_client, processed_tweet_ids)

            db_analysis_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "db_analysis.json")
            if db_analysis_blob_client.exists():
                existing_content = db_analysis_blob_client.download_blob().readall().decode('utf-8')
                existing_data = json.loads(existing_content)
                combined_data = combine_json_data(combined_data_chunk, existing_data)
            else:
                combined_data = combined_data_chunk

            app.logger.info(f"Combined data before cleaning: {combined_data}")

            # Filter zero-values and pretty print the JSON
            combined_data_cleaned = {entity: {category: {k: v for k, v in details.items() if v != 0} for category, details in data.items()} for entity, data in combined_data.items()}
            
            app.logger.info(f"Cleaned data: {combined_data_cleaned}")

            combined_json_content = json.dumps(combined_data_cleaned, indent=4)

            blob_upload_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "db_analysis.json")
            blob_upload_client.upload_blob(combined_json_content, overwrite=True)
            
            update_processed_tweet_ids(blob_service_client, new_processed_ids_chunk + list(processed_tweet_ids))
            processed_tweet_ids.update(new_processed_ids_chunk)

        return jsonify({'message': 'Data processed successfully'}), 200

    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({'message': f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run()
