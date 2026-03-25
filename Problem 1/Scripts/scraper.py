import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import time
import re

def is_valid_url(url, base_domain):
    """Check if the URL belongs to the base domain and is a web page."""
    parsed = urlparse(url)
    # Ensure it's an HTTP/HTTPS link and belongs to iitj.ac.in
    if parsed.scheme not in ('http', 'https'):
        return False
    if base_domain not in parsed.netloc:
        return False
    # Avoid downloading files directly during the html scrape
    invalid_extensions = ('.pdf', '.doc', '.docx', '.png', '.jpg', '.zip')
    if parsed.path.lower().endswith(invalid_extensions):
        return False
    return True

def clean_text(text):
    """Basic removal of excessive newlines and spaces."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def bfs_scrape_iitj(start_url, max_pages=50, output_file="iitj_corpus.txt"):
    """
    Performs a BFS crawl of the given URL.
    max_pages is strictly set to avoid infinite loops or overloading the server.
    """
    base_domain = "iitj.ac.in"
    queue = deque([start_url])
    visited = set([start_url])
    
    pages_scraped = 0
    
    # Open the output file in append mode
    with open(output_file, 'a', encoding='utf-8') as f:
        
        while queue and pages_scraped < max_pages:
            current_url = queue.popleft()
            print(f"Scraping [{pages_scraped + 1}/{max_pages}]: {current_url}")
            
            try:
                # Add a small delay to be polite to the server
                time.sleep(1) 
                response = requests.get(current_url, timeout=10)
                
                # Only process successful HTML responses
                if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 1. Extract Text (Targeting paragraphs and headings)
                    text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])
                    page_text = " ".join([elem.get_text() for elem in text_elements])
                    cleaned_page_text = clean_text(page_text)
                    
                    if cleaned_page_text:
                        f.write(cleaned_page_text + "\n\n")
                    
                    # 2. Find new links and add to queue
                    for link in soup.find_all('a', href=True):
                        absolute_link = urljoin(current_url, link['href'])
                        # Remove URL fragments (the part after #)
                        absolute_link = urlparse(absolute_link)._replace(fragment="").geturl()
                        
                        if absolute_link not in visited and is_valid_url(absolute_link, base_domain):
                            visited.add(absolute_link)
                            queue.append(absolute_link)
                            
            except requests.RequestException as e:
                print(f"Failed to retrieve {current_url}: {e}")
            
            pages_scraped += 1

    print(f"\nScraping complete. Text saved to {output_file}.")

if __name__ == "__main__":
    # Start at the main page, or a specific department page
    start_url = "https://iitj.ac.in/"
    # You can increase max_pages, but keep it reasonable to avoid getting blocked
    bfs_scrape_iitj(start_url, max_pages=100)