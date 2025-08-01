import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_website(url):
    # Add headers to mimic browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Send GET request to the URL
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Example: Extract all paragraph texts
        paragraphs = soup.find_all('p')
        texts = [p.text for p in paragraphs]
        
        # Example: Extract all links
        links = soup.find_all('a')
        urls = [link.get('href') for link in links]
        
        # Create a dictionary to store the data
        data = {
            'texts': texts,
            'urls': urls
        }
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv('scraped_data.csv', index=False)
        print("Data successfully scraped and saved to 'scraped_data.csv'")
        
    except requests.RequestException as e:
        print(f"Error occurred while scraping: {e}")
    
    # Add delay to be respectful to the server
    time.sleep(2)

if __name__ == "__main__":
    # Replace with the URL you want to scrape
    target_url = "https://example.com"
    scrape_website(target_url)
