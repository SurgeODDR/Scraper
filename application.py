import signal
import sys
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

# Configure logging for Azure Log Stream
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

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
    headers = {
        "Authorization": f"Bearer {openai.api_key}"
    }
    
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [
            {"role": "system", "content": """
You are analyzing the text provided. Provide a quantitative analysis in CSV format. The analysis should cover:
- Distribution of sentiments (Positive, Negative, Neutral) with percentages.
- Distribution of key emotions (happiness, sadness, anger, fear, surprise, disgust, jealousy, outrage/indignation, distrust/skepticism, despair/hopelessness, shock/astonishment, relief, and empowerment) with percentages.
- Mentions of keywords related to inequality, unfairness, diminished trust in the government, unjust actions, disloyalty, and perceptions of corruption, and their associated sentiment percentages.
- Highlight and quantify mentions of well-known figures such as celebrities, politicians, or other public figures and analyze the sentiment and emotional tones associated with these mentions.

Structure the CSV output as follows:
"Category, Positive (%), Negative (%), Neutral (%), Total Mentions"
"Sentiments, [Positive Percentage], [Negative Percentage], [Neutral Percentage], [Total Sentiment Mentions]"
"Emotions: Happiness, [Positive Percentage], -, -, [Total Happiness Mentions]"
"Emotions: Sadness, [Negative Percentage], -, -, [Total Sadness Mentions]"
...
"Keywords: Inequality, -, [Negative Percentage], -, [Total Inequality Mentions]"
"Keywords: Corruption, -, [Negative Percentage], -, [Total Corruption Mentions]"
...
"Public Figures: [Public Figure Name], [Positive Percentage], [Negative Percentage], [Neutral Percentage], [Total Mention of the Figure]"
"""
            },
            {"role": "user", "content": text}
        ],
        "temperature": 0.3,
        "max_tokens": 16000
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    response_data = response.json()

    if 'choices' in response_data:
        app.logger.info(f"Analysis took {time.time() - start_time} seconds")
        return response_data['choices'][0]['message']['content'].strip()
    else:
        app.logger.error(f"Unexpected response from OpenAI: {response_data}")
        return "Error analyzing the text."

@app.route('/process', methods=['GET'])
def process_data():
    try:
        blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
        
        # Fetch the data from Tweets.json
        blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
        download_stream = blob_client.download_blob()
        data = download_stream.readall()
        app.logger.info(f"Data from blob: {data[:100]}")
        df = pd.read_json(io.BytesIO(data))
        
        # Split the data into chunks for analysis
        chunk_size = 10  # Adjust as necessary
        chunks = [df['text'][i:i+chunk_size].str.cat(sep='\n') for i in range(0, len(df), chunk_size)]
        
        # Initialize the tweets_processed variable
        tweets_processed = 0
        
        # Analyze each chunk and update aggregate analysis
        for i, chunk in enumerate(chunks):
            # Analyze the chunk
            analysis = analyze_text(chunk)
            tweets_processed += len(chunk.split('\n'))  # Count the number of tweets in the current chunk
            
            # Update the aggregate analysis with this analysis and the number of tweets processed
            update_aggregate_analysis(analysis, tweets_processed)
        
        return jsonify({'message': 'Data processed and aggregate analysis updated successfully'}), 200

    except AzureError as ae:
        app.logger.error(f"AzureError encountered: {str(ae)}")
    except Exception as e:
        app.logger.error(f"General error encountered: {str(e)}")
    
    return jsonify({'error': 'An error occurred while processing the data.'}), 500
        
def summarize_chunk(chunk_text):
    """Summarizes a chunk of text using OpenAI."""
    headers = {
        "Authorization": f"Bearer {openai.api_key}"
    }
    
    tweet_count = len(chunk_text.split('\n'))
    
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [
            {
                "role": "system",
                "content": f""""
You are analyzing a set of {tweet_count} tweets. Provide a quantitative summary in CSV format. The summary should contain:
- How many Tweets are in this dataset
- Distribution of sentiments (Positive, Negative, Neutral) with percentages.
- Distribution of key emotions (Anger, Distrust, Skepticism, Outrage/Indignation) with percentages.
- Total mentions of keywords related to inequality and corruption, and their associated sentiment percentages.

Structure the CSV output as follows:
"Category, Positive (%), Negative (%), Neutral (%), Total Mentions"
"Sentiments, x%, y%, z%, Total"
"Emotions: Anger, a%, -, -, Total"
"Emotions: Distrust, b%, -, -, Total"
...
"Keywords: Inequality, -, p%, -, Total"
"Keywords: Corruption, -, q%, -, Total"
...
"""
            },
            {"role": "user", "content": chunk_text}
        ],
        "temperature": 0.3,
        "max_tokens": 12000
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    response_data = response.json()
    
    if 'choices' in response_data:
        return response_data['choices'][0]['message']['content'].strip()
    else:
        app.logger.error(f"Unexpected response from OpenAI: {response_data}")
        return "Error summarizing the data."
        
import time

def update_aggregate_analysis(summary_chunk, tweets_processed):
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
            {"role": "system", "content": f"You are tasked with updating the aggregate analysis with the new summary provided. As of now, {tweets_processed} tweets have been analyzed. Ensure the update adheres to academic standards. Integrate the new summary into the existing aggregate analysis in a way that maintains a cohesive, comprehensive, and academically robust narrative."},
            {"role": "user", "content": aggregate_text + "\n\n" + summary_chunk}
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
       

@app.route('/summarize', methods=['GET'])
def summarize_data():
    try:
        # Initialize the BlobServiceClient
        blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
        
        # Get the blob client for the consolidated JSON file
        blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Analysed_Tweets.json")

        # Download the JSON file
        download_stream = blob_client.download_blob()
        df = pd.read_json(io.BytesIO(download_stream.readall()))

        # Split the data into chunks for summarization
        chunk_size = 10
        chunks = [df['Analysis'][i:i+chunk_size].str.cat(sep='\n') for i in range(0, len(df), chunk_size)]

        # Summarize each chunk and update aggregate analysis
        tweets_processed = 0
        for i, chunk in enumerate(chunks):
            summary_blob_name = f"Summary_Chunk_{i}.txt"
            summary_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", summary_blob_name)
            
            if not summary_blob_client.exists():
                # Summarize the chunk
                summary = summarize_chunk(chunk)
                tweets_processed += len(chunk.split('\n'))  # Count the number of tweets in the current chunk
                
                # Store the summary
                summary_blob_client.upload_blob(summary, overwrite=True)
                # Update the aggregate analysis with this summary and the number of tweets processed
                update_aggregate_analysis(summary, tweets_processed)
        
        # After all chunks are processed, fetch the final aggregate analysis
        with open("/tmp/aggregate_analysis.txt", 'r') as file:
            final_summary = file.read()

        # Create a Document instance and add the final summary to it
        doc = Document()
        doc.add_paragraph(final_summary)

        # Save the document to a .docx file
        doc_path = '/tmp/Final_Summary.docx'
        doc.save(doc_path)

        # Upload the .docx file to Azure Blob Storage
        blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Final_Summary.docx")
        with open(doc_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)

        return jsonify({'message': 'Final summary created and saved successfully'}), 200

    except Exception as e:
        app.logger.error(f"Error in /summarize route: {e}")
        return jsonify({'error': 'An error occurred while summarizing the data.'}), 500


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
