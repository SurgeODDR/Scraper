import io
from flask import Flask, request, jsonify
import pandas as pd
import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os
import textwrap
from azure.storage.blob import BlobServiceClient
import logging

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

    app.logger.info('Processing data...')

    try:
        blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
        blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
        download_stream = blob_client.download_blob().readall()
        df = pd.read_json(io.StringIO(download_stream.decode('utf-8')))
        df = df[['text']]
        df['Analysis'] = df['text'].apply(analyze_text)
        df.to_json('Analysed_Tweets.json', orient='records')
        app.logger.info('Data processed successfully.')
        return jsonify({'message': 'Data processed successfully'}), 200
    except Exception as e:
        app.logger.error(f'Error processing data: {e}')
        return jsonify({'error': 'Error processing data'}), 500

if __name__ == "__main__":
    app.run(debug=True)
