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
    """Performs topic modeling, sentiment analysis, and emotional tone analysis using GPT-3."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a research assistant analyzing public sentiment towards the Panama Papers tax scandal. Analyze the following text for key themes, sentiment, and emotional tone:"},
            {"role": "user", "content": text}
        ],
        temperature=0.3,
        max_tokens=200
    )
    return response['choices'][0]['message']['content'].strip()

@app.route('/process', methods=['GET'])
def process_data():
    """Fetches the data from Azure Storage and performs the analysis."""
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
    download_stream = blob_client.download_blob()
    df = pd.read_json(io.BytesIO(download_stream.readall()))

    chunk_size = 100  # Adjust this value based on your needs
    num_chunks = len(df) // chunk_size
    if len(df) % chunk_size:
        num_chunks += 1

    app.logger.info(f"Processing {num_chunks} chunks")

    for i in range(num_chunks):
        app.logger.info(f"Processing chunk {i + 1} of {num_chunks}")
        try:
            chunk_blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", f"Analysed_Tweets_{i}.json")
            if chunk_blob_client.exists():
                app.logger.info(f"Chunk {i + 1} already processed, skipping")
                continue

            start = i * chunk_size
            end = start + chunk_size
            df_chunk = df[start:end]
            df_chunk['Analysis'] = df_chunk['text'].apply(analyze_text)
            chunk_json = df_chunk.to_json(orient='records')
            chunk_blob_client.upload_blob(chunk_json, overwrite=True)

            app.logger.info(f"Finished processing chunk {i + 1}")
            time.sleep(5)  # Add a delay to prevent overloading the server
        except Exception as e:
            app.logger.error(f"Error processing chunk {i + 1}: {str(e)}")

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

@app.route('/summarize', methods=['GET'])
def summarize_data():
    """Fetches the analysis results from Azure Storage and generates a summary."""
    global df

    # Create a BlobServiceClient using the credential
    logging.info("Fetching analysed data from Azure Storage...")
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)

    # Get the blob client for the JSON file
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Analysed_Tweets.json")

    # Download the JSON file
    download_stream = blob_client.download_blob().content_as_text()

    # Load the data into a pandas DataFrame
    df = pd.read_json(io.StringIO(download_stream))

    # Generate a summary of the analysis results
    logging.info("Generating summary of analysis results...")
    summary = summarize_results(df['Analysis'].to_string())

    # Create a Document instance
    doc = Document()

    # Add the summary to the document
    doc.add_paragraph(summary)

    # Save the document to a .docx file
    doc.save('/tmp/Summary.docx')

    # Create a BlobClient for the .docx file
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Summary.docx")

    # Upload the .docx file to the blob
    logging.info("Uploading summary to Azure Storage...")
    with open('/tmp/Summary.docx', 'rb') as data:
        blob_client.upload_blob(data)

    return jsonify({'message': 'Summary created and saved successfully'}), 200

if __name__ == "__main__":
    app.run(debug=True)
