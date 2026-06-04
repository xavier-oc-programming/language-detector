from pathlib import Path

MODEL_DIR = Path('models')
PLOTS_DIR = Path('plots')
RANDOM_STATE = 42
TEST_SIZE = 0.2

# character n-grams from unigrams to trigrams — more robust than word-level
# because they capture script patterns, morphological patterns, and character
# combinations that are language-specific regardless of vocabulary.
# A trigram like 'lar' is a strong Turkish suffix. 'the' is English.
# 'que' appears in French and Spanish but with different surrounding patterns.
NGRAM_RANGE = (1, 3)

# 'char_wb' treats word boundaries explicitly — n-grams do not span word
# boundaries. Word-boundary patterns (how words start and end) are highly
# language-specific, making this better than plain 'char' for language detection.
ANALYZER = 'char_wb'

MAX_FEATURES = 50000
TOP_N_LANGUAGES = 3
TOP_N_NGRAMS = 5

AZURE_APP_NAME = 'language-detector-xoc'
AZURE_RESOURCE_GROUP = 'language-detector-rg'
AZURE_LOCATION = 'westeurope'
