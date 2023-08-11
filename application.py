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

# Fetch the second OpenAI API key
openai_api_key2 = secret_client.get_secret("openai-api-key2").value

# Create a separate rate limiter for the second API key
rate_limiter2 = RateLimiter(max_calls=3500, period=60)

def openai_request(data, api_key, rate_limiter_obj):
    """Make a rate-limited request to the OpenAI API."""
    headers = {"Authorization": f"Bearer {api_key}"}
    
    with rate_limiter_obj:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        return response.json()
        
def save_new_analysis(blob_service_client, analysis):
    new_analysis_path = "/tmp/new_analysis.txt"
    with open(new_analysis_path, 'w') as file:
        file.write(analysis)
    new_analysis_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "new_analysis.txt")
    with open(new_analysis_path, 'rb') as data:
        new_analysis_blob_client.upload_blob(data, overwrite=True)     
        
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
...
"""
            },
            {"role": "user", "content": text}
        ],
        "temperature": 0.3,
        "max_tokens": 12000
    }
    
    response_data = openai_request(data, openai.api_key, rate_limiter)

    if 'choices' in response_data:
        return response_data['choices'][0]['message']['content'].strip()
    else:
        print("Error in OpenAI response.")
        return "Error analyzing the text."

def update_processed_tweet_ids(blob_service_client, processed_ids):
    """Update the blob with new processed tweet IDs."""
    logging.info("Starting update of processed tweet IDs...")
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "processed_tweet_ids.txt")
    ids_str = "\n".join(map(str, processed_ids))
    blob_client.upload_blob(ids_str, overwrite=True)
    logging.info(f"Successfully updated processed_tweet_ids.txt with {len(processed_ids)} IDs")

@app.route('/process', methods=['GET'])
def process_data():
    print("Processing data...")

    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
    download_stream = blob_client.download_blob()
    data = download_stream.readall()
    df = pd.read_json(io.BytesIO(data))

    processed_ids = get_processed_tweet_ids(blob_service_client)
    print(f"Previously processed tweet IDs: {processed_ids}")  # Debugging log

    # Filter out tweets that have already been processed
    df = df[~df['id'].isin(processed_ids)]

    chunk_size = 5
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    new_processed_ids = set()

    for chunk_df in chunks:
        chunk_text = chunk_df['text'].str.cat(sep='\n')
        analysis = analyze_text(chunk_text)
        new_processed_ids.update(chunk_df['id'].tolist())
        update_aggregate_analysis(blob_service_client, analysis, len(chunk_df))

    # Add the newly processed IDs to the existing set and save them
    processed_ids.update(new_processed_ids)
    update_processed_tweet_ids(blob_service_client, processed_ids)

    print(f"Updated processed tweet IDs: {processed_ids}")  # Debugging log
    return jsonify({'message': 'Data processed successfully'}), 200
    
def compare_files(blob_service_client):
    print("Comparing aggregate analysis files...")
    aggregate_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "aggregate_analysis.txt")
    now_aggregate_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "now_aggregate_analysis.txt")

    if aggregate_blob_client.exists():
        aggregate_content = aggregate_blob_client.download_blob().readall().decode('utf-8')
    else:
        aggregate_content = ""  # Handle the case where the blob doesn't exist yet

    now_aggregate_content = now_aggregate_blob_client.download_blob().readall().decode('utf-8')

    headers = {"Authorization": f"Bearer {openai_api_key2}"}
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [
            {
                "role": "system",
                "content": "Are the values bigger or the same in 'now_aggregate_analysis.txt' compared to 'aggregate_analysis.txt'? The response can only be YES or NO."
            },
            {
                "role": "user",
                "content": f"Old Aggregate Analysis:\n{aggregate_content}\n\nNew Aggregate Analysis:\n{now_aggregate_content}"
            }
        ],
        "temperature": 0.3,
        "max_tokens": 12000
    }

    response = openai_request(data, openai_api_key2, rate_limiter2)  # Use the second API key and its rate limiter
    response_data = response

    if 'choices' in response and response['choices'][0]['message']['content'].strip() == "YES":
        print("now_aggregate_analysis.txt has bigger or the same values as aggregate_analysis.txt")
        return True
    print("now_aggregate_analysis.txt does NOT have bigger or the same values as aggregate_analysis.txt")
    return False
    
def get_processed_tweet_ids(blob_service_client):
    """Fetch the list of tweet IDs that have been processed."""
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "processed_tweet_ids.txt")
    
    if blob_client.exists():
        download_stream = blob_client.download_blob()
        processed_ids = set(map(int, download_stream.readall().decode('utf-8').splitlines()))
        return processed_ids
    
    return set()

def update_aggregate_analysis(blob_service_client, analysis, tweets_processed):
    print("Updating aggregate analysis...")

    # Save the new analysis to new_analysis.txt
    save_new_analysis(blob_service_client, analysis)
    
    aggregate_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "aggregate_analysis.txt")
    new_analysis_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "new_analysis.txt")
    
    aggregate_content = ""  # Initialize the variable here
    iteration = 1
    if aggregate_blob_client.exists():
        aggregate_content = aggregate_blob_client.download_blob().readall().decode('utf-8')
        last_line = aggregate_content.strip().split('\n')[-1]
        if "Iteration" in last_line:
            iteration = int(last_line.split(" ")[1].replace(":", "")) + 1
    else:
        # If aggregate_analysis.txt does not exist, create it with the content of now_aggregate_analysis.txt
        now_aggregate_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "now_aggregate_analysis.txt")
        if now_aggregate_blob_client.exists():
            now_aggregate_content = now_aggregate_blob_client.download_blob().readall()
            aggregate_blob_client.upload_blob(now_aggregate_content, overwrite=True)

    new_analysis_content = new_analysis_blob_client.download_blob().readall().decode('utf-8')
    
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [
            {
                "role": "system",
                "content": """
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
                "content": f"Existing Aggregate Analysis:\n{aggregate_content}\n\nNew Analysis:\n{new_analysis_content}"
            }
        ],
        "temperature": 0.3,
        "max_tokens": 12000
    }
    
    response_data = openai_request(data, openai.api_key, rate_limiter)
    if 'choices' in response_data:
        combined_content = response_data['choices'][0]['message']['content'].strip() + f"\n\n---\nIteration: {iteration} | Updated on: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        now_aggregate_path = "/tmp/now_aggregate_analysis.txt"
        with open(now_aggregate_path, 'w') as file:
            file.write(combined_content)
        
        now_aggregate_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "now_aggregate_analysis.txt")
        with open(now_aggregate_path, 'rb') as data:
            now_aggregate_blob_client.upload_blob(data, overwrite=True)
    
        is_valid = compare_files(blob_service_client)
        if is_valid:
            with open(now_aggregate_path, 'rb') as data:
                aggregate_blob_client.upload_blob(data, overwrite=True)
            print("Updated aggregate_analysis.txt successfully")
        else:
            print("The new aggregate analysis did not have bigger or the same values as the previous one.")

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
