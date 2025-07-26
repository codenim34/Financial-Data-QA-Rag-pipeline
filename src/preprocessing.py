"""
Preprocessing module for financial data RAG pipeline.
Handles PDF text extraction, cleaning, and chunking.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import pdfplumber
import PyPDF2
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)


class PDFPreprocessor:
    """Handles PDF text extraction and preprocessing."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the PDF preprocessor.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.stop_words = set(stopwords.words('english'))
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using multiple methods for better coverage.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        text = ""
        
        # Method 1: Using pdfplumber (better for tables and structured content)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            logger.info(f"Successfully extracted text using pdfplumber from {pdf_path}")
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
            
        # Method 2: Using PyPDF2 as fallback
        if not text.strip():
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                logger.info(f"Successfully extracted text using PyPDF2 from {pdf_path}")
            except Exception as e:
                logger.error(f"PyPDF2 also failed: {e}")
                raise ValueError(f"Could not extract text from {pdf_path}")
        
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\,\-\$\%\(\)]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'\b\d+\s*of\s*\d+\b', '', text)
        
        return text.strip()
    
    