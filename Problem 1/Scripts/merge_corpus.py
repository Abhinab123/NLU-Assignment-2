import os
import fitz  # PyMuPDF
import re

def clean_pdf_text(text):
    """Cleans extracted PDF text by removing extra whitespaces, newlines, and non-ascii artifacts."""
    # Replace multiple spaces and newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    # Optional: Strip out non-ASCII characters to keep the vocabulary clean for Word2Vec
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def merge_pdfs_to_corpus(pdf_directory, existing_corpus_file, final_corpus_file):
    """
    Extracts text from PDFs, merges it with the web-scraped corpus, 
    and saves it to a final text file.
    """
    print(f"Creating final corpus: {final_corpus_file}")
    
    # Open the final corpus file in write mode
    with open(final_corpus_file, 'w', encoding='utf-8') as outfile:
        
        # 1. Copy the existing web-scraped text first
        if os.path.exists(existing_corpus_file):
            print(f"Merging web-scraped data from {existing_corpus_file}...")
            with open(existing_corpus_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
            outfile.write("\n\n") 
        else:
            print(f"Warning: {existing_corpus_file} not found. Starting fresh with only PDFs.")

        # 2. Process and append PDF text
        if not os.path.exists(pdf_directory):
            print(f"Directory '{pdf_directory}' not found. Please create it and add your PDFs.")
            return

        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in '{pdf_directory}'.")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            print(f"Extracting text from: {pdf_file}...")
            
            try:
                # Open the PDF using PyMuPDF
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text("text")
                    
                    cleaned_text = clean_pdf_text(text)
                    if cleaned_text:
                        # Write the cleaned page text to the final corpus
                        outfile.write(cleaned_text + "\n")
                
                outfile.write("\n\n") # Add separation between documents
                doc.close()
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")

    print("\nCorpus merging complete! Your data is ready in:", final_corpus_file)

if __name__ == "__main__":
    # --- Configuration ---
    # Create a folder named 'pdfs' in your project directory and put your files there
    PDF_DIR = "pdfs"  
    WEB_CORPUS = "iitj_corpus.txt" 
    FINAL_CORPUS = "corpus.txt" 
    
    merge_pdfs_to_corpus(PDF_DIR, WEB_CORPUS, FINAL_CORPUS)