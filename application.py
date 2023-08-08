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
            {
                "role": "user",
                "content": ("You are tasked with an analysis of sentiment and emotion within the provided "
                            "textual content, in accordance with academic standards. The examination is divided into two parts:\n\n"
                            "1. **Sentiment Quantification:**\n"
                            "- **Quantification:** Determine the sentiment present in the text, providing a percentage distribution "
                            "for positive, negative, and neutral sentiments.\n"
                            "- **Thematic Analysis:** Identify mentions of inequality, unfairness, diminished trust in government, "
                            "perceptions of corruption, unjust actions, or disloyalty.\n"
                            "- **Interpretation:** Explain how these sentiments align with public opinions, especially if connected "
                            "to well-known individuals such as celebrities or politicians.\n\n"
                            "2. **Emotion Analysis:**\n"
                            "- **Identification and Quantification:** Detect and quantify key emotions in the text, including "
                            "happiness, sadness, anger, fear, surprise, disgust, jealousy, outrage/indignation, distrust/skepticism, "
                            "despair/hopelessness, shock/astonishment, relief, and empowerment.\n"
                            "- **Thematic Emphasis:** Comment on sentiments indicating feelings of inequality, unfairness, or perceptions "
                            "of corruption.\n"
                            "- **Contextual Interpretation:** Provide an interpretation of these emotional tones in the context of "
                            "significant events or subjects, such as the Panama Papers.")
            }
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

@app.route('/summarize', methods=['GET'])
def summarize_data():
    """Fetches the analysis results from Azure Storage and generates a summary."""
    global df

    if df is None:
        return jsonify({'error': 'No data available'}), 400

    # Fetch the blob service client from the Azure Storage
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    
    # Get the blob client for the consolidated JSON file
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Analysed_Tweets.json")

    # Download the JSON file
    download_stream = blob_client.download_blob()
    df = pd.read_json(io.BytesIO(download_stream.readall()))

    # Generate a summary of the analysis results
    summary = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a research assistant who has analyzed public sentiment towards the Panama Papers tax scandal using topic modeling, sentiment analysis, and emotional tone analysis. Your task is to create an academic summary of the findings. Here are the analysis results:"},
            {"role": "user", "content": df['Analysis'].to_string()}
        ],
        temperature=0.3,
        max_tokens=12000
    )['choices'][0]['message']['content'].strip()

    # Create a Document instance
    doc = Document()

    # Add the summary to the document
    doc.add_paragraph(summary)

    # Save the document to a .docx file
    doc.save('/tmp/Summary.docx')

    # Create a BlobClient instance
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Summary.docx")

    # Upload the .docx file to the blob
    with open('/tmp/Summary.docx', 'rb') as data:
        blob_client.upload_blob(data)

    return jsonify({'message': 'Summary created and saved successfully'}), 200

if __name__ == "__main__":
    app.run(debug=True)
