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
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            app.logger.error(f"Error in OpenAI API request: {e}")
            return {}
        
def save_new_analysis(blob_service_client, analysis):
    try:
        new_analysis_path = "/tmp/new_analysis.txt"
        with open(new_analysis_path, 'w') as file:
            file.write(analysis)
        new_analysis_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "new_analysis.txt")
        with open(new_analysis_path, 'rb') as data:
            new_analysis_blob_client.upload_blob(data, overwrite=True) 
    except Exception as e:
        app.logger.error(f"Error saving new analysis to blob: {e}")   
        
def analyze_text(text):
    app.logger.info("Analyzing text...")
    headers = {"Authorization": f"Bearer {openai.api_key}"}
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [
            {"role": "system", "content": """
    Generate a quantitative analysis in CSV format based on the provided text. Cover:
    - Sentiments (Positive, Negative, Neutral)
    - Key emotions (happiness, sadness, anger, fear, surprise, disgust, jealousy, outrage/indignation, distrust/skepticism, despair/hopelessness, shock/astonishment, relief, and empowerment)
    - Keywords related to inequality, unfairness, distrust in government, unjust actions, disloyalty, and perceptions of corruption.
    
    CSV Structure:
    "Category, Positive (%), Negative (%), Neutral (%), Total Mentions"
    "Sentiments, [Positive Percentage], [Negative Percentage], [Neutral Percentage], [Total Sentiment Mentions]"
    "Emotions: Happiness, [Positive Percentage], -, -, [Total Happiness Mentions]"
    ...
    "Keywords: Inequality, -, [Negative Percentage], -, [Total Inequality Mentions]"
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
        app.logger.info("Error in OpenAI response.")
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
    app.logger.info("Processing data...")
    
    try:
        blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
        blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
        download_stream = blob_client.download_blob()
        data = download_stream.readall()
        df = pd.read_json(io.BytesIO(data))
    except Exception as e:
        app.logger.error(f"Error fetching or processing tweets data from blob: {e}")
        return jsonify({'message': 'Error fetching or processing tweets data'}), 500

    try:
        processed_ids = get_processed_tweet_ids(blob_service_client)
        app.logger.info(f"Previously processed tweet IDs: {processed_ids}")  # Debugging log
        
        # Filter out tweets that have already been processed
        df = df[~df['id'].isin(processed_ids)]

        chunk_size = 5
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        for chunk_df in chunks:
            chunk_text = chunk_df['text'].str.cat(sep='\n')
            analysis = analyze_text(chunk_text)
            
            # Add the newly processed IDs to the existing set and save them immediately after processing the chunk
            processed_ids.update(chunk_df['id'].tolist())
            update_processed_tweet_ids(blob_service_client, processed_ids)
            
            update_aggregate_analysis(blob_service_client, analysis, len(chunk_df))

        app.logger.info(f"Updated processed tweet IDs: {processed_ids}")  # Debugging log
        return jsonify({'message': 'Data processed successfully'}), 200
    except Exception as e:
        app.logger.error(f"Error processing data: {e}")
        return jsonify({'message': 'Error during data processing'}), 500
    
def compare_files(blob_service_client):
    app.logger.info("Comparing aggregate analysis files...")
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
                "content": """
    Please compare each corresponding value in 'now_aggregate_analysis.txt' against 'aggregate_analysis.txt'. For every category and sub-category, ensure that the value in 'now_aggregate_analysis.txt' is either the same or larger than the corresponding value in 'aggregate_analysis.txt'. Respond with 'YES' if all values in 'now_aggregate_analysis.txt' meet this criterion, otherwise respond with 'NO'.
    Example:
    "Sentiments, 50%, 30%, 20%, 100" in 'aggregate_analysis.txt' should be compared with "Sentiments, 55%, 28%, 17%, 105" in 'now_aggregate_analysis.txt'. The latter has higher or equal values for all categories.
    """
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
        app.logger.info("now_aggregate_analysis.txt has bigger or the same values as aggregate_analysis.txt")
        return True
    app.logger.info("now_aggregate_analysis.txt does NOT have bigger or the same values as aggregate_analysis.txt")
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
    app.logger.info("Updating aggregate analysis...")

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
            {"role": "system", "content": """
    Merge the provided existing aggregate analysis with the new analysis into a single cohesive dataset. Ensure the merged data is in CSV format. Combine similar data points without duplication and ensure accuracy.
    """
            },
            {"role": "user", "content": f"Existing Aggregate Analysis:\n{aggregate_content}\n\nNew Analysis:\n{new_analysis_content}"}
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
            app.logger.info("Updated aggregate_analysis.txt successfully")
        else:
            app.logger.info("The new aggregate analysis did not have bigger or the same values as the previous one.")

FLAG_TRIGGER_PROCESS = True

def monitor_and_trigger_process():
    while True:
        if FLAG_TRIGGER_PROCESS:
            try:
                requests.get("https://gptanalyser.azurewebsites.net/process")
                FLAG_TRIGGER_PROCESS = False
            except Exception as e:
                app.logger.info(f"Error: {e}")
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
            app.logger.info(f"Exception: {e}")
            process.terminate()
            monitor_process.terminate()
            process = Process(target=run_app)
            process.start()
            monitor_process = Process(target=monitor_and_trigger_process)
            monitor_process.start()
