import requests
import os
import json
import time
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.core.exceptions import ResourceNotFoundError
from requests.exceptions import HTTPError
import logging

class TwitterAPI:
    def __init__(self, bearer_token, account_url):
        self.bearer_token = bearer_token
        self.credential = DefaultAzureCredential()
        self.blob_service_client = BlobServiceClient(account_url=account_url, credential=self.credential)

    def create_url(self, ids):
        tweet_fields = "tweet.fields=lang,author_id,text,created_at"
        ids = "ids=" + ",".join(ids)
        url = f"https://api.twitter.com/2/tweets?{ids}&{tweet_fields}"
        return url

    def bearer_oauth(self, r):
        r.headers["Authorization"] = f"Bearer {self.bearer_token}"
        r.headers["User-Agent"] = "v2TweetLookupPython"
        return r

    def connect_to_endpoint(self, url):
        logging.info(f"Sending request to: {url}")
        response = requests.request("GET", url, auth=self.bearer_oauth)
        
        if response.status_code != 200:
            logging.error(f"Request returned an error: {response.status_code}, {response.text}")
            return None
        return response.json()

    def get_blob_text(self, container_name, blob_name):
        try:
            logging.info(f"Reading blob: {blob_name}")
            blob_client = self.blob_service_client.get_blob_client(container_name, blob_name)
            return blob_client.download_blob().readall().decode('utf-8')
        except ResourceNotFoundError as e:
            logging.error(f"Blob not found: {blob_name}", exc_info=True)
            raise e

    def upload_blob_text(self, container_name, blob_name, content):
        logging.info(f"Uploading blob: {blob_name}")
        blob_client = self.blob_service_client.get_blob_client(container_name, blob_name)
        blob_client.upload_blob(content, overwrite=True)

    def get_tweet_ids_from_file(self, container_name, blob_name):
        ids_text = self.get_blob_text(container_name, blob_name)
        return ids_text.strip().split('\n')

    def main(self, container_name, tweet_ids_blob_name, output_blob_name, last_tweet_id_blob_name, failed_tweet_ids_blob_name):
        logging.info("Fetching tweet IDs")
        tweet_ids = self.get_tweet_ids_from_file(container_name, tweet_ids_blob_name)

        try:
            last_tweet_id = self.get_blob_text(container_name, last_tweet_id_blob_name).strip()
            tweet_ids = tweet_ids[tweet_ids.index(last_tweet_id) + 1:]
        except Exception as e:
            logging.error("Error reading last_tweet_id_blob", exc_info=True)

        try:
            failed_tweet_ids = set(self.get_blob_text(container_name, failed_tweet_ids_blob_name).strip().split('\n'))
            tweet_ids = [id for id in tweet_ids if id not in failed_tweet_ids]
        except Exception as e:
            logging.error("Error reading failed_tweet_ids_blob", exc_info=True)
            failed_tweet_ids = set()

        chunks = [tweet_ids[i:i + 100] for i in range(0, len(tweet_ids), 100)]
        all_tweets = []
        for chunk in chunks:
            url = self.create_url(chunk)
            json_response = self.connect_to_endpoint(url)
            if json_response is not None:
                all_tweets.extend(json_response.get('data', []))
                self.upload_blob_text(container_name, last_tweet_id_blob_name, chunk[-1])
            else:
                failed_tweet_ids.update(chunk)
                self.upload_blob_text(container_name, failed_tweet_ids_blob_name, '\n'.join(failed_tweet_ids))
                logging.info("Rate limit reached. Waiting 15 minutes before retrying...")
                time.sleep(15 * 60)

            self.upload_blob_text(container_name, output_blob_name, json.dumps(all_tweets, indent=4, sort_keys=True))
            logging.info(f"Data saved to blob: {output_blob_name}")

def run_app():
    logging.info("Initializing DefaultAzureCredential")
    credential = DefaultAzureCredential()

    logging.info("Initializing SecretClient")
    secret_client = SecretClient(vault_url="https://Keyvaultxscrapingoddr.vault.azure.net", credential=credential)

    logging.info("Fetching secrets from Key Vault")
    BEARER_TOKEN = secret_client.get_secret("bearer-token").value
    ACCOUNT_URL = "https://scrapingstoragex.blob.core.windows.net"

    logging.info("Initializing TwitterAPI")
    twitter_api = TwitterAPI(BEARER_TOKEN, ACCOUNT_URL)
    
    logging.info("Running TwitterAPI")
    twitter_api.main('scrapingstoragecontainer', 'tweetids.txt', 'Tweets.json', 'last_tweet_id.txt', 'failed_tweet_ids.txt')

if __name__ == "__main__":
    run_app()
