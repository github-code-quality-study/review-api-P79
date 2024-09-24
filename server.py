import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            query_params = parse_qs(environ.get("QUERY_STRING", ""))
            location = query_params.get('location', [None])[0]
            start_date_str = query_params.get('start_date', [None])[0]
            end_date_str = query_params.get('end_date', [None])[0]

            filtered_reviews = reviews

            if location:
                filtered_reviews = [review for review in filtered_reviews if review.get('Location') == location]

            if start_date_str:
                # Convert start_date_str to a datetime object
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                # Filter reviews based on the start date
                filtered_reviews = [
                    review for review in filtered_reviews
                    if datetime.strptime(review['Timestamp'], TIMESTAMP_FORMAT) >= start_date
                ]

            if end_date_str:
                # Convert end_date_str to a datetime object
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                # Filter reviews based on the end date
                filtered_reviews = [
                    review for review in filtered_reviews
                    if datetime.strptime(review['Timestamp'], TIMESTAMP_FORMAT) <= end_date
                ]

            # Analyze sentiment for each review and add it to the review data
            for review in filtered_reviews:
                review_body = review.get('ReviewBody', '')
                review['sentiment'] = self.analyze_sentiment(review_body)

            # Sort reviews by sentiment's compound score in descending order
            sorted_reviews = sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'], reverse=True)

            # Create the response body from the sorted reviews and convert to a JSON byte string
            response_body = json.dumps(sorted_reviews, indent=2).encode("utf-8")         
            
            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
                      
           try:
                content_length = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(content_length).decode('utf-8')
                review_data = parse_qs(request_body)  # Use parse_qs to handle form-encoded data

                # Extract values from the parsed data
                location = review_data.get('Location', [None])[0]
                review_body = review_data.get('ReviewBody', [None])[0]

                # Check for missing location or review body
                if not location or not review_body:
                    raise ValueError("Location and ReviewBody are required")
                
                # Optionally, validate the location here
                # Example validation (assuming you have a set of valid locations)
                VALID_LOCATIONS = {'San Diego, California', 'New York, New York'}
                if location not in VALID_LOCATIONS:
                    raise ValueError("Invalid location")
                
                # Process the review data
                review_data = {
                    'Location': location,
                    'ReviewBody': review_body,
                    'ReviewId': str(uuid.uuid4()),
                    'Timestamp': datetime.now().strftime(TIMESTAMP_FORMAT)
                }

                    # Append the new review and save to CSV
                reviews.append(review_data)
                pd.DataFrame(reviews).to_csv('data/reviews.csv', index=False)

                response_body = json.dumps(review_data, indent=2).encode("utf-8")
                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

           except ValueError as e:
                response_body = json.dumps({"error": str(e)}).encode("utf-8")
                start_response("400 Bad Request", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
                ])
                return [response_body]
           except Exception as e:
                response_body = json.dumps({"error": "An unexpected error occurred: " + str(e)}).encode("utf-8")
                start_response("500 Internal Server Error", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]
          
if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()

