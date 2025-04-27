
# Import necessary libraries
from textblob import TextBlob

# Sample hypothetical data (e.g., tweets or reviews)
data = [
    "I love this product, it works amazing!",
    "This is the worst service I have ever received.",
    "I feel okay about the event, neither good nor bad.",
    "Fantastic experience! Will definitely come back again.",
    "Very disappointing, I expected better quality."
]

# Function to analyze sentiment using TextBlob
def analyze_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)
    
    # Get the sentiment polarity
    polarity = blob.sentiment.polarity
    
    # Categorize the sentiment based on polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis on each piece of data
results = []
for text in data:
    sentiment = analyze_sentiment(text)
    results.append((text, sentiment))

# Print the results
for text, sentiment in results:
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
    print()
