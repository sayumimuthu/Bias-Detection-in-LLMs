#!/bin/bash

# Install required dependencies for NLP-based protagonist attribute extraction
echo "Installing NLP dependencies for bias detection..."

# Install base packages
pip install spacy textblob nltk pandas numpy matplotlib seaborn scipy

# Download spaCy model
echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('punkt')"

# Download TextBlob corpora
echo "Downloading TextBlob corpora..."
python -m textblob.download_corpora

echo "✓ All dependencies installed successfully!"
echo "You can now run: python protagonist_attributes_nlp.py"
