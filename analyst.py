from flask import Flask, request, jsonify
import pandas as pd
import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os
import textwrap
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

def sentiment_analysis(text):
    """Performs sentiment analysis on the provided text using the GPT-3 model."""
    chunks = textwrap.wrap(text, width=3000)
    analysis_results = []
    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "Use your capabilities as an AI assistant to conduct a sentiment analysis on the following text, considering the context of tax scandals and public opinion:"},
                {"role": "user", "content": chunk}
            ],
            temperature=0.3,
            max_tokens=12000
        )
        analysis_results.append(response['choices'][0]['message']['content'].strip())
    return " ".join(analysis_results)

@app.route('/process', methods=['GET'])
def process_data():
    """Fetches the data from Azure Storage and performs sentiment analysis."""
    global df
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
    download_stream = blob_client.download_blob()
    df = pd.read_json(download_stream.readall())
    df = df[['text']]
    df['Sentiment_Analysis'] = df['text'].apply(sentiment_analysis)
    df.to_json('Analysed_Tweets.json', orient='records')
    return jsonify({'message': 'Data processed successfully'}), 200

@app.route('/results', methods=['GET'])
def get_results():
    """Returns the sentiment analysis results."""
    global df
    if df is None:
        return jsonify({'error': 'No data available'}), 400
    return df.to_json(orient='records'), 200

if __name__ == "__main__":
    app.run(debug=True)
