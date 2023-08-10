import signal
from multiprocessing import Process
from flask import Flask, request, jsonify
import pandas as pd
import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
import os
import io
import json
import logging
import time
from docx import Document
from azure.core.exceptions import AzureError
import requests

# Create a Flask instance
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Initialize a global variable for the DataFrame
df = None

# Authenticate using the managed identity
credential = DefaultAzureCredential()

# Create a SecretClient using the credential
key_vault_uri = "https://Keyvaultxscrapingoddr.vault.azure.net/"
secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)

# Get the OpenAI API key from the Key Vault
openai.api_key = secret_client.get_secret("openai-api-key").value

def analyze_text(text):
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a research assistant specializing in sentiment and emotion analysis. Analyze the following text, adhering to academic standards:\n\nSentiment Quantification: Quantify the sentiment expressed within the text, providing a percentage distribution of positive, negative, and neutral sentiments. Include an interpretation of how these sentiments align with public opinion on the subject. Investigate any mentions of inequality, unfairness, diminished trust in the government, unjust actions, disloyalty, and perceptions of corruption.\n\nEmotion Analysis: Identify and quantify the presence of key emotions such as happiness, sadness, anger, fear, surprise, disgust, jealousy, outrage/indignation, distrust/skepticism, despair/hopelessness, shock/astonishment, relief, and empowerment within the text. Provide an interpretation of these emotional tones in the context of the subject matter, such as the Panama Papers.\n\nSpecial Attention: Highlight any mentions of well-known figures such as celebrities, politicians, or other public figures, and analyze the sentiment and emotional tones associated with these mentions."},
            {"role": "user", "content": text}
        ],
        temperature=0.3,
        max_tokens=16000  # Reduced max tokens to speed up processing
    )
    app.logger.info(f"Analysis took {time.time() - start_time} seconds")
    return response['choices'][0]['message']['content'].strip()

@app.route('/process', methods=['GET'])
def process_data():
    try:
        blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
        blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
        download_stream = blob_client.download_blob()
        data = download_stream.readall()
        app.logger.info(f"Data from blob: {data[:100]}")
        df = pd.read_json(io.BytesIO(data))
        chunk_size = 3  # Reduced chunk size for testing
        num_chunks = len(df) // chunk_size
        if len(df) % chunk_size:
            num_chunks += 1
        app.logger.info(f"Processing {num_chunks} chunks")

        for i in range(num_chunks):
            app.logger.info(f"Processing chunk {i + 1} of {num_chunks}")
            chunk_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", f"Analysed_Tweets_{i}.json")
            if chunk_blob_client.exists():
                app.logger.info(f"Chunk {i + 1} already processed, skipping")
                continue
            start = i * chunk_size
            end = start + chunk_size
            df_chunk = df[start:end].copy()
            app.logger.info(f"Analyzing chunk {i + 1}")
            df_chunk['Analysis'] = df_chunk['text'].apply(analyze_text)
            chunk_json = df_chunk.to_json(orient='records')
            app.logger.info(f"Uploading chunk {i + 1} to blob")
            chunk_blob_client.upload_blob(chunk_json, overwrite=True)
            app.logger.info(f"Finished processing chunk {i + 1}, saved to blob")
            time.sleep(5)  # This delay helps to avoid hitting any API limits
    except AzureError as ae:
        app.logger.error(f"AzureError encountered: {str(ae)}")
    except Exception as e:
        app.logger.error(f"General error encountered: {str(e)}")
    
    return jsonify({'message': 'Data processed successfully'}), 200

@app.route('/consolidate', methods=['GET'])
def consolidate_data():
    """Fetches the chunked data from Azure Storage and consolidates it into a single file."""
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    container_client = blob_service_client.get_container_client("scrapingstoragecontainer")

    df_consolidated = pd.DataFrame()

    for blob in container_client.list_blobs():
        if "Analysed_Tweets_" in blob.name:
            app.logger.info(f"Consolidating {blob.name}")
            blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", blob.name)
            download_stream = blob_client.download_blob()
            df_chunk = pd.read_json(io.BytesIO(download_stream.readall()))
            df_consolidated = pd.concat([df_consolidated, df_chunk])

    consolidated_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Analysed_Tweets.json")
    consolidated_blob_client.upload_blob(df_consolidated.to_json(orient='records'), overwrite=True)

    return jsonify({'message': 'Data consolidated successfully'}), 200

def summarize_chunk(chunk_text):
    """Summarizes a chunk of text using OpenAI."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a research assistant specializing in sentiment and emotion analysis of public reactions to major events. Summarize the following analysis results related to the Panama Papers scandal."},
            {"role": "user", "content": chunk_text}
        ],
        temperature=0.3,
        max_tokens=12000
    )
    return response['choices'][0]['message']['content'].strip()

@app.route('/summarize', methods=['GET'])
def summarize_data():
    # Initialize the BlobServiceClient
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    
    # Get the blob client for the consolidated JSON file
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Analysed_Tweets.json")

    # Download the JSON file
    download_stream = blob_client.download_blob()
    df = pd.read_json(io.BytesIO(download_stream.readall()))

    # Split the data into chunks for summarization
    chunk_size = 500
    chunks = [df['Analysis'][i:i+chunk_size].str.cat(sep='\n') for i in range(0, len(df), chunk_size)]

    # Summarize each chunk, but first check if the summary for that chunk already exists
    summaries = []
    for i, chunk in enumerate(chunks):
        summary_blob_name = f"Summary_Chunk_{i}.txt"
        summary_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", summary_blob_name)
        
        if summary_blob_client.exists():
            # Fetch the existing summary for this chunk
            download_stream = summary_blob_client.download_blob()
            summary = download_stream.readall().decode('utf-8')
        else:
            # Summarize the chunk and store the summary
            summary = summarize_chunk(chunk)
            summary_blob_client.upload_blob(summary, overwrite=True)
        
        summaries.append(summary)

    # Combine the individual summaries to get a final summary
    final_summary = " ".join(summaries)

    # Create a Document instance
    doc = Document()

    # Add the final summary to the document
    doc.add_paragraph(final_summary)

    # Save the document to a .docx file
    doc.save('/tmp/Summary.docx')

    # Create a BlobClient instance
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Summary.docx")

    # Upload the .docx file to the blob
    with open('/tmp/Summary.docx', 'rb') as data:
        blob_client.upload_blob(data)

    return jsonify({'message': 'Summary created and saved successfully'}), 200


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
