import requests
from flask import Flask, jsonify
import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import io
import logging
from ratelimiter import RateLimiter
from azure.keyvault.secrets import SecretClient

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

credential = DefaultAzureCredential()
key_vault_uri = "https://Keyvaultxscrapingoddr.vault.azure.net/"
secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)
openai_api_key = secret_client.get_secret("openai-api-key").value

rate_limiter = RateLimiter(max_calls=3500, period=60)

def openai_request(data):
    """Make a rate-limited request to the OpenAI API."""
    headers = {"Authorization": f"Bearer {openai_api_key}"}
    
    with rate_limiter:
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            app.logger.error(f"Error in OpenAI API request: {e}")
            return {}

def append_analysis_to_blob(blob_service_client, analysis):
    """Append an individual analysis to the text file on the blob."""
    try:
        blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "individual_analyses.txt")
        
        if blob_client.exists():
            current_content = blob_client.download_blob().readall().decode('utf-8')
        else:
            current_content = ""

        combined_content = current_content + "\n" + analysis
        blob_client.upload_blob(combined_content, overwrite=True)
    except Exception as e:
        app.logger.error(f"Error appending analysis to blob: {e}")

def analyze_text(text):
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [
            {"role": "system", "content": """
        Generate a quantitative analysis in CSV format based on the provided text. Cover:
        - Sentiments (Positive, Negative, Neutral)
        - Key emotions (happiness, sadness, anger, fear, surprise, disgust, jealousy, outrage/indignation, distrust/skepticism, despair/hopelessness, shock/astonishment, relief, and empowerment)
        - Keywords related to inequality, unfairness, distrust in government, unjust actions, disloyalty, and perceptions of corruption.
        """
            },
            {"role": "user", "content": text}
        ],
        "temperature": 0.3,
        "max_tokens": 12000
    }

    response_data = openai_request(data)
    return response_data['choices'][0]['message']['content'].strip() if 'choices' in response_data else "Error analyzing the text."

@app.route('/process', methods=['GET'])
def process_data():
    blob_service_client = BlobServiceClient(account_url="https://scrapingstoragex.blob.core.windows.net", credential=credential)
    blob_client = blob_service_client.get_blob_client("scrapingstoragecontainer", "Tweets.json")
    
    try:
        download_stream = blob_client.download_blob()
        data = download_stream.readall()
        df = pd.read_json(io.BytesIO(data))
    except Exception as e:
        app.logger.error(f"Error fetching or processing tweets data from blob: {e}")
        return jsonify({'message': 'Error fetching or processing tweets data'}), 500
    
    for _, row in df.iterrows():
        tweet_text = row['text']
        analysis = analyze_text(tweet_text)
        append_analysis_to_blob(blob_service_client, analysis)
        
    return jsonify({'message': 'Data processed successfully'}), 200

if __name__ == "__main__":
    app.run(debug=True)
