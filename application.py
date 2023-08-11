import requests
from flask import Flask, jsonify
import pandas as pd
import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
import io
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

rate_limiter = RateLimiter(max_calls=3500, period=60)

def openai_request(data):
    """Make a rate-limited request to the OpenAI API."""
    headers = {"Authorization": f"Bearer {openai.api_key}"}
    
    with rate_limiter:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        return response.json()
        
def analyze_text(text):
    print("Analyzing text...")
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
    
    response_data = openai_request(data)

    if 'choices' in response_data:
        return response_data['choices'][0]['message']['content'].strip()
    else:
        print("Error in OpenAI response.")
        return "Error analyzing the text."

def get_last_processed_tweet_id(blob_service_client):
    print("Fetching last processed tweet ID...")
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "last_processed_id.txt")
    if blob_client.exists():
        download_stream = blob_client.download_blob()
        return int(download_stream.readall())
    return None

def set_last_processed_tweet_id(blob_service_client, tweet_id):
    print(f"Setting last processed tweet ID: {tweet_id}...")
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "last_processed_id.txt")
    blob_client.upload_blob(str(tweet_id), overwrite=True)

@app.route('/process', methods=['GET'])
def process_data():
    print("Processing data...")
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
    download_stream = blob_client.download_blob()
    data = download_stream.readall()
    df = pd.read_json(io.BytesIO(data))

    last_processed_id = get_last_processed_tweet_id(blob_service_client)
    if last_processed_id:
        df = df[df['id'] > last_processed_id]

    chunk_size = 5
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

    for chunk_df in chunks:
        chunk_text = chunk_df['text'].str.cat(sep='\n')
        analysis = analyze_text(chunk_text)
        last_tweet_id_in_chunk = chunk_df.iloc[-1]['id']
        set_last_processed_tweet_id(blob_service_client, last_tweet_id_in_chunk)
        update_aggregate_analysis(analysis, len(chunk_df))

    return jsonify({'message': 'Data processed successfully'}), 200

def update_aggregate_analysis(analysis, tweets_processed):
    print("Updating aggregate analysis...")
    aggregate_path = "/tmp/aggregate_analysis.txt"
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    aggregate_text = ""
    iteration = 1
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "aggregate_analysis.txt")
    
    if blob_client.exists():
        download_stream = blob_client.download_blob()
        aggregate_text = download_stream.readall().decode('utf-8')
        last_line = aggregate_text.strip().split('\n')[-1]
        if "Iteration" in last_line:
            iteration = int(last_line.split(" ")[1].replace(":", "")) + 1
    
    headers = {"Authorization": f"Bearer {openai.api_key}"}
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
    
    response_data = openai_request(data)

    if 'choices' in response_data:
        updated_text = response_data['choices'][0]['message']['content'].strip() + f"\n\n---\nIteration: {iteration} | Updated on: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        with open(aggregate_path, 'w') as file:
            file.write(updated_text)
        with open(aggregate_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)

FLAG_TRIGGER_PROCESS = True

def monitor_and_trigger_process():
    while True:
        if FLAG_TRIGGER_PROCESS:
            try:
                requests.get("https://gptanalyser.azurewebsites.net/process")
                FLAG_TRIGGER_PROCESS = False
            except Exception as e:
                print(f"Error: {e}")
            time.sleep(60)

def run_app():
    app.run(debug=True)

if __name__ == "__main__":
    from multiprocessing import Process
    process = Process(target=run_app)
    process.start()
    monitor_process = Process(target=monitor_and_trigger_process)
    monitor_process.start()

    while True:
        try:
            process.join(timeout=60)
            if not process.is_alive():
                break
        except (KeyboardInterrupt, SystemExit):
            process.terminate()
            monitor_process.terminate()
            process.join()
            monitor_process.join()
            break
        except Exception as e:
            print(f"Exception: {e}")
            process.terminate()
            monitor_process.terminate()
            process = Process(target=run_app)
            process.start()
            monitor_process = Process(target=monitor_and_trigger_process)
            monitor_process.start()
