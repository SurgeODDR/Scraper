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
                    It's crucial that the output is in a precise CSV format with the correct headers. Analyze the text to identify mentions of celebrities or politicians. Determine the associated sentiments (Positive, Negative, Neutral) and emotions (happiness, sadness, anger, fear, surprise, disgust, jealousy, outrage, distrust, despair, shock, relief, empowerment) for each mentioned figure. Present the results in this specific CSV format, aggregating counts for each sentiment or emotion per figure.
                    
                    Desired CSV structure (including headers):
                    "Celebrity/Politician Name, Sentiment/Emotion, Total Mentions"
                    For example:
                    "John Doe, Sentiments: Positive, 5"
                    "Jane Smith, Emotions: Anger, 3"
                    ...
                    
                    Ensure the headers are present in the output.
                """
            },
            {"role": "user", "content": text}
        ],
        "temperature": 0.1,
        "max_tokens": 13000
    }

    response_data = openai_request(data)  # Using the function without unnecessary arguments
    if 'choices' in response_data:
        return response_data['choices'][0]['message']['content'].strip()
    return "Error analyzing the text."

def combine_and_save_analysis(blob_service_client, new_analysis):
    try:
        new_df = pd.read_csv(io.StringIO(new_analysis))
        
        # Ensure necessary columns are present in the new dataframe
        if 'Celebrity/Politician Name' not in new_df.columns:
            app.logger.error("Expected column 'Celebrity/Politician Name' not found in new analysis.")
            return

        celeb_db_analysis_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "celeb_db_analysis.csv")
        if celeb_db_analysis_blob_client.exists():
            existing_content = celeb_db_analysis_blob_client.download_blob().readall().decode('utf-8')
            existing_df = pd.read_csv(io.StringIO(existing_content))
            
            # Concatenate the new and existing dataframes
            combined_df = pd.concat([new_df, existing_df])
            
            # Group by name and sentiment/emotion and then sum the total mentions
            combined_df = combined_df.groupby(['Celebrity/Politician Name', 'Sentiment/Emotion']).sum().reset_index()
        else:
            combined_df = new_df

        combined_csv_content = combined_df.to_csv(index=False)
        save_to_blob(blob_service_client, combined_csv_content, "celeb_db_analysis.csv")

    except Exception as e:
        app.logger.error(f"Error in combine_and_save_analysis: {str(e)}")

@app.route('/process', methods=['GET'])
def process_data():
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
    df = pd.read_json(io.BytesIO(blob_client.download_blob().readall()))

    processed_ids = get_processed_tweet_ids(blob_service_client)
    df = df[~df['id'].isin(processed_ids)]
    
    chunk_size = 5
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    for chunk_df in chunks:
        chunk_text = chunk_df['text'].str.cat(sep='\n')
        new_analysis = analyze_text(chunk_text)
        combine_and_save_analysis(blob_service_client, new_analysis)
        processed_ids.update(chunk_df['id'].tolist())

    save_to_blob(blob_service_client, "\n".join(map(str, processed_ids)), "processed_tweet_ids.txt")
    return jsonify({'message': 'Data processed successfully'}), 200

if __name__ == '__main__':
    app.run()
