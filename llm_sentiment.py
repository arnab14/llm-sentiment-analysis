import csv
import requests
import json
import time # Import time for potential delays/retries

# --- Configuration ---
CSV_INPUT_FILE = 'reviews.csv'
CSV_OUTPUT_FILE = 'reviews_with_llm_sentiment.csv'
OLLAMA_API_URL = 'http://localhost:11434/api/generate' # Default Ollama API endpoint
OLLAMA_MODEL = 'gemma3:27b' # Or 'llama3:8b', 'mistral', etc. - Make sure it's downloaded!
REQUEST_TIMEOUT = 60 # Timeout for the API request in seconds

# --- Prompt Template ---
# Instructs the LLM to provide only one word: Positive, Negative, or Neutral.
PROMPT_TEMPLATE = """
Analyze the sentiment of the following product review.
Respond with only one word: Positive, Negative, or Neutral.

Review: "{review_text}"

Sentiment:"""

# --- Function to Interact with Ollama ---
def get_llm_sentiment(review_text, retries=3, delay=5):
    """
    Sends the review text to the Ollama API for sentiment analysis.

    Args:
        review_text (str): The product review text.
        retries (int): Number of times to retry the request if it fails.
        delay (int): Seconds to wait between retries.

    Returns:
        str: The predicted sentiment ('Positive', 'Negative', 'Neutral', or 'Error').
    """
    prompt = PROMPT_TEMPLATE.format(review_text=review_text)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False # We want the full response at once
    }
    headers = {'Content-Type': 'application/json'}

    for attempt in range(retries):
        try:
            print(f"  Sending review to Ollama (Attempt {attempt + 1}/{retries})...")
            response = requests.post(
                OLLAMA_API_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            response_data = response.json()
            llm_response_text = response_data.get('response', '').strip()

            # --- Basic Parsing and Validation ---
            # Convert to title case to handle variations like 'positive' or 'POSITIVE'
            llm_response_text = llm_response_text.title()

            if llm_response_text in ['Positive', 'Negative', 'Neutral']:
                print(f"  Ollama responded: {llm_response_text}")
                return llm_response_text
            else:
                # Handle cases where the LLM didn't follow instructions exactly
                print(f"  Warning: Unexpected LLM response format: '{llm_response_text}'. Attempting to find keyword.")
                # Simple check if the expected keywords are present
                if 'Positive' in llm_response_text: return 'Positive'
                if 'Negative' in llm_response_text: return 'Negative'
                if 'Neutral' in llm_response_text: return 'Neutral'
                print("  Error: Could not reliably parse sentiment from LLM response.")
                return 'Error: Parse Failed' # Indicate a parsing failure

        except requests.exceptions.RequestException as e:
            print(f"  Error contacting Ollama API (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                print(f"  Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("  Max retries reached. Skipping this review.")
                return 'Error: API Failed' # Indicate an API failure

    return 'Error: Max Retries' # Should not typically be reached if loop exits normally


# --- Main Processing Logic ---
results = []

print(f"Starting sentiment analysis using Ollama model: {OLLAMA_MODEL}")

try:
    with open(CSV_INPUT_FILE, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        if 'ProductID' not in reader.fieldnames or 'ReviewText' not in reader.fieldnames:
             raise ValueError("CSV file must contain 'ProductID' and 'ReviewText' columns.")

        for i, row in enumerate(reader):
            product_id = row['ProductID']
            review_text = row['ReviewText']
            print(f"\nProcessing review {i+1} (ProductID: {product_id})...")
            print(f"  Review: \"{review_text[:100]}...\"") # Print truncated review

            sentiment = get_llm_sentiment(review_text)

            results.append({
                'ProductID': product_id,
                'ReviewText': review_text,
                'Sentiment': sentiment
            })

except FileNotFoundError:
    print(f"Error: Input file '{CSV_INPUT_FILE}' not found.")
    exit()
except ValueError as ve:
     print(f"Error in CSV format: {ve}")
     exit()
except Exception as e:
    print(f"An unexpected error occurred during file reading: {e}")
    exit()


# --- Output Results ---
print("\n--- Analysis Complete ---")
print("Results:")
for result in results:
    print(f"ProductID: {result['ProductID']}, Sentiment: {result['Sentiment']}")
    # Uncomment below to print the full review text as well
    # print(f"  Review: {result['ReviewText']}")

# --- Bonus: Save to CSV ---
if results:
    try:
        with open(CSV_OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as outfile:
            fieldnames = ['ProductID', 'ReviewText', 'Sentiment']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults successfully saved to '{CSV_OUTPUT_FILE}'")
    except IOError as e:
        print(f"\nError: Could not write results to CSV file '{CSV_OUTPUT_FILE}': {e}")
else:
    print("\nNo results to save.")

