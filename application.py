from flask import Flask, request, jsonify
import pandas as pd
import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os
import textwrap
import io
from azure.storage.blob import BlobServiceClient
import numpy as np

# Create a Flask instance
app = Flask(__name__)

# Initialize a global variable for the DataFrame
df = None

# Authenticate using the managed identity
credential = DefaultAzureCredential()

# Create a SecretClient using the credential
key_vault_uri = "https://Keyvaultxscrapingoddr.vault.azure.net/"
secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)

# Get the OpenAI API key from the Key Vault
openai.api_key = secret_client.get_secret("openai-api-key").value

def chunker(seq, size):
    """Splits the data into chunks."""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def analyze_text_in_chunks(df, chunk_size=10):
    """Performs topic modeling, sentiment analysis, and emotional tone analysis using GPT-3 on chunks of data."""
    df['Analysis'] = np.nan
    for i, chunk in enumerate(chunker(df, chunk_size)):
        text = ' '.join(chunk['text'])
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a research assistant analyzing public sentiment towards the Panama Papers tax scandal. Analyze the following text for key themes, sentiment, and emotional tone:"},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=16000 - len(text) // 4  # adjust the number of output tokens to account for the input length
        )
        df.loc[chunk.index, 'Analysis'] = response['choices'][0]['message']['content'].strip()
    return df

@app.route('/process', methods=['GET'])
def process_data():
    """Fetches the data from Azure Storage and performs the analysis."""
    global df
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
    download_stream = blob_client.download_blob()
    df = pd.read_json(io.BytesIO(download_stream.readall()))
    df = df[['text']]
    df = analyze_text_in_chunks(df)
    df.to_json('Analysed_Tweets.json', orient='records')
    return jsonify({'message': 'Data processed successfully'}), 200

@app.route('/summarize', methods=['GET'])
def summarize_analysis():
    """Summarizes the analysis results."""
    global df
    if df is None or 'Analysis' not in df.columns:
        return jsonify({'error': 'No analysis data available'}), 400
    summary = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a research assistant who has just completed an analysis of public sentiment towards the Panama Papers tax scandal using GPT-3.5 Turbo. Please create an academic summary of the results."},
            {"role": "user", "content": ' '.join(df['Analysis'].dropna())}
        ],
        temperature=0.3,
        max_tokens=4096
    )
    return jsonify({'summary': summary['choices'][0]['message']['content'].strip()}), 200

if __name__ == "__main__":
    app.run(debug=True)
