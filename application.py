import logging
from flask import Flask, request, jsonify
import pandas as pd
import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
import os
import textwrap
from docx import Document
import io

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Create a Flask instance
app = Flask(__name__)

# Initialize a global variable for the DataFrame
df = None

# Authenticate using the managed identity
logging.info("Authenticating with Azure...")
credential = DefaultAzureCredential()

# Create a SecretClient using the credential
logging.info("Creating SecretClient...")
key_vault_uri = "https://Keyvaultxscrapingoddr.vault.azure.net/"
secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)

# Get the OpenAI API key from the Key Vault
logging.info("Fetching OpenAI API key...")
openai.api_key = secret_client.get_secret("openai-api-key").value

def analyze_text(text):
    """Performs sentiment analysis using GPT-3."""
    logging.info("Analyzing text...")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a research assistant analyzing public sentiment towards the Panama Papers tax scandal. Analyze the following text for key themes, sentiment, and emotional tone:"},
            {"role": "user", "content": text}
        ],
        temperature=0.3,
        max_tokens=6000
    )
    return response['choices'][0]['message']['content'].strip()

@app.route('/process', methods=['GET'])
def process_data():
    """Fetches the data from Azure Storage and performs the analysis."""
    global df
    logging.info("Fetching data from Azure Storage...")
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
    download_stream = blob_client.download_blob().content_as_text()
    df = pd.read_json(io.StringIO(download_stream))
    df = df[['text']]
    logging.info("Performing sentiment analysis on data...")
    df['Analysis'] = df['text'].apply(analyze_text)
    df.to_json('Analysed_Tweets.json', orient='records')
    return jsonify({'message': 'Data processed successfully'}), 200

def summarize_results(data):
    """Summarizes the analysis results using GPT-3."""
    logging.info("Summarizing analysis results...")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a research assistant who has analyzed public sentiment towards the Panama Papers tax scandal using topic modeling, sentiment analysis, and emotional tone analysis. Your task is to create an academic summary of the findings."},
            {"role": "user", "content": data}
        ],
        temperature=0.3,
        max_tokens=12000
    )
    return response['choices'][0]['message']['content'].strip()

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
