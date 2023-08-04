import io
import os
import textwrap
import numpy as np
import pandas as pd
import openai
from flask import Flask, jsonify
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

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

def analyze_text(text):
    """Performs sentiment analysis using GPT-3."""
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

    # Connect to Azure Storage and download the data
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
    download_stream = blob_client.download_blob().readall()

    # Load the data into a DataFrame
    df = pd.read_json(io.StringIO(download_stream.decode('utf-8')))

    # Only keep the 'text' column
    df = df[['text']]

    # Create a new column for chunk IDs
    df['Chunk'] = np.arange(len(df)) // 100

    # Apply sentiment analysis to each chunk separately
    for _, chunk_df in df.groupby('Chunk'):
        chunk_df['Analysis'] = chunk_df['text'].apply(analyze_text)

    # Drop the 'Chunk' column
    df = df.drop(columns=['Chunk'])

    # Save the DataFrame to a JSON file
    df.to_json('Analysed_Tweets.json', orient='records')

    return jsonify({'message': 'Data processed successfully'}), 200

if __name__ == "__main__":
    app.run(debug=True)
