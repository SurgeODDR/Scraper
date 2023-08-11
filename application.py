import requests
from flask import Flask, jsonify
import pandas as pd
import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
import os
import io
import json
import sys
import logging
import time
from ratelimiter import RateLimiter

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

credential = DefaultAzureCredential()
key_vault_uri = "https://Keyvaultxscrapingoddr.vault.azure.net/"
secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)
openai.api_key = secret_client.get_secret("openai-api-key").value

rate_limiter = RateLimiter(max_calls=3500, period=60)  # 3,500 requests per minute

def analyze_text(text):
    headers = {"Authorization": f"Bearer {openai.api_key}"}
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [
            {"role": "system", "content": """
You are analyzing the text provided. Provide a quantitative analysis in CSV format. The analysis should cover:
- Distribution of sentiments (Positive, Negative, Neutral) with percentages.
- Distribution of key emotions (happiness, sadness, anger, fear, surprise, disgust, jealousy, outrage/indignation, distrust/skepticism, despair/hopelessness, shock/astonishment, relief, and empowerment) with percentages.
- Mentions of keywords related to inequality, unfairness, diminished trust in the government, unjust actions, disloyalty, and perceptions of corruption, and their associated sentiment percentages.

Structure the CSV output as follows:
"Category, Positive (%), Negative (%), Neutral (%), Total Mentions"
"Sentiments, [Positive Percentage], [Negative Percentage], [Neutral Percentage], [Total Sentiment Mentions]"
"Emotions: Happiness, [Positive Percentage], -, -, [Total Happiness Mentions]"
"Emotions: Sadness, [Negative Percentage], -, -, [Total Sadness Mentions]"
...
"Keywords: Inequality, -, [Negative Percentage], -, [Total Inequality Mentions]"
"Keywords: Corruption, -, [Negative Percentage], -, [Total Corruption Mentions]"
"""
            },
            {"role": "user", "content": text}
        ],
        "temperature": 0.3,
        "max_tokens": 12000
    }
    
    with rate_limiter:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response_data = response.json()

    if 'error' in response_data and response_data['error']['code'] == 'rate_limit_exceeded':
        time.sleep(1)
        return analyze_text(text)

    if 'choices' in response_data:
        return response_data['choices'][0]['message']['content'].strip()
    else:
        app.logger.error(f"Unexpected response from OpenAI: {response_data}")
        return "Error analyzing the text."

LAST_PROCESSED_ID_BLOB_NAME = "last_processed_id.txt"

def get_last_processed_tweet_id(blob_service_client):
    """Fetch the last processed tweet ID from Azure Blob Storage."""
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", LAST_PROCESSED_ID_BLOB_NAME)
    if blob_client.exists():
        download_stream = blob_client.download_blob()
        return int(download_stream.readall())
    return None

def set_last_processed_tweet_id(blob_service_client, tweet_id):
    """Save the last processed tweet ID to Azure Blob Storage."""
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", LAST_PROCESSED_ID_BLOB_NAME)
    blob_client.upload_blob(str(tweet_id), overwrite=True)

@app.route('/process', methods=['GET'])
def process_data():
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
    download_stream = blob_client.download_blob()
    data = download_stream.readall()
    df = pd.read_json(io.BytesIO(data))

    # Fetch the last processed tweet ID and use it as a starting point
    last_processed_id = get_last_processed_tweet_id(blob_service_client)
    if last_processed_id:
        df = df[df['id'] > last_processed_id]

    chunk_size = 5
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

    for chunk_df in chunks:
        chunk_text = chunk_df['text'].str.cat(sep='\n')
        analysis = analyze_text(chunk_text)
        
        # Save the ID of the last tweet in the chunk to Azure Blob Storage
        last_tweet_id_in_chunk = chunk_df.iloc[-1]['id']
        set_last_processed_tweet_id(blob_service_client, last_tweet_id_in_chunk)

        # Here's where you should update the aggregate analysis
        update_aggregate_analysis(analysis, len(chunk_df))

    return jsonify({'message': 'Data processed and aggregate analysis updated successfully'}), 200

        
import time

def update_aggregate_analysis(analysis, tweets_processed):
    """Updates the aggregate_analysis.txt with the summary_chunk and uploads it to Azure Blob Storage."""

    aggregate_path = "/tmp/aggregate_analysis.txt"
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    
    # Default values
    aggregate_text = ""
    iteration = 1
    
    # Check if the aggregate_analysis.txt file exists in Blob Storage
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "aggregate_analysis.txt")
    if blob_client.exists():
        # Download the file from Blob Storage
        download_stream = blob_client.download_blob()
        with open(aggregate_path, 'wb') as file:
            file.write(download_stream.readall())
        with open(aggregate_path, 'r') as file:
            aggregate_text = file.read()
            # Extract the last iteration number from the aggregate text
            last_line = aggregate_text.strip().split('\n')[-1]
            if "Iteration" in last_line:
                iteration = int(last_line.split(" ")[1].replace(":", "")) + 1
    
    headers = {
        "Authorization": f"Bearer {openai.api_key}"
    }

    # Modify the prompt to include the number of tweets processed
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [
            {
                "role": "system",
                "content": f"""
    You have two sets of data: an existing aggregate analysis and a new analysis. Your task is to integrate the new analysis into the existing one. Follow these steps:
    1. Ensure that the integrated data is in CSV format.
    2. Maintain academic rigor and standards throughout the process.
    3. Keep the narrative cohesive and comprehensive.
    4. If there are similar data points between the new analysis and the existing aggregate, combine them accurately without duplication.
    5. Ensure that the final integrated data is accurate and reflects the true nature of both the existing and new analyses.
    Remember, the integrity and accuracy of the data are paramount.
    """
            },
            {
                "role": "user",
                "content": f"Existing Aggregate Analysis:\n{aggregate_text}\n\nNew Analysis:\n{analysis}"
            }
        ],
        "temperature": 0.3,
        "max_tokens": 12000
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    response_data = response.json()

    if 'choices' in response_data:
        updated_text = response_data['choices'][0]['message']['content'].strip()
        
        # Append the iteration number and timestamp to the updated text
        updated_text += f"\n\n---\nIteration: {iteration} | Updated on: {time.strftime('%Y-%m-%d %H:%M:%S')}"

        # Write the updated text to the aggregate_analysis.txt
        with open(aggregate_path, 'w') as file:
            file.write(updated_text)

        # Upload the updated aggregate_analysis.txt to Azure Blob Storage
        with open(aggregate_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)
        logging.info(f"Uploaded {aggregate_path} to Azure Blob Storage successfully")

    else:
        error_message = f"Unexpected response from OpenAI: {response_data}"
        app.logger.error(error_message)
        logging.error(error_message)
       

FLAG_TRIGGER_PROCESS = True  # Set this flag to True to trigger /process

def monitor_and_trigger_process():
    global FLAG_TRIGGER_PROCESS
    while True:
        if FLAG_TRIGGER_PROCESS:
            try:
                # Send a request to the /process endpoint
                response = requests.get("https://gptanalyser.azurewebsites.net/process")
                if response.status_code == 200:
                    FLAG_TRIGGER_PROCESS = False  # Reset the flag after successful processing
            except Exception as e:
                app.logger.error(f"Failed to trigger /process: {e}")
            time.sleep(60)  # Check every minute

def run_app():
    app.run(debug=True)

if __name__ == "__main__":
    process = Process(target=run_app)
    process.start()

    # Start the monitoring process
    monitor_process = Process(target=monitor_and_trigger_process)
    monitor_process.start()

    while True:
        try:
            # Check every minute if the Flask process is still alive
            process.join(timeout=60)
            if not process.is_alive():
                break
        except (KeyboardInterrupt, SystemExit):
            # Gracefully shutdown
            process.terminate()
            monitor_process.terminate()
            process.join()
            monitor_process.join()
            break
        except Exception as e:
            # Log the error for debugging purposes
            app.logger.error(f"Exception encountered: {e}")

            # Try to gracefully terminate the processes
            process.terminate()
            monitor_process.terminate()
            process.join()
            monitor_process.join()

            # Restart the Flask app in a new process
            process = Process(target=run_app)
            process.start()

            # Restart the monitoring process
            monitor_process = Process(target=monitor_and_trigger_process)
            monitor_process.start()
