import datetime
import logging
import azure.functions as func
from .app import run_app  # Import the run_app function from app.py

def main(mytimer: func.TimerRequest) -> None:
    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()

    if mytimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Python timer trigger function ran at %s', utc_timestamp)
    
    run_app()  # Call the run_app function when the timer triggers