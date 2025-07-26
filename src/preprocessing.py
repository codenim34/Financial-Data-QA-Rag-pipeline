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



    """
    Convenience function for complete financial data preprocessing.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Size of text chunks
        
    Returns:
        Processed data dictionary
    """
    preprocessor = PDFPreprocessor(chunk_size=chunk_size)
    return preprocessor.process_pdf(pdf_path) 