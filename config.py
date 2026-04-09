"""
Configuration file
"""

import os
from pathlib import Path

# API Keys - Please fill in your API keys here
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-350e5068abb745919baa79e2673ce763")
JINA_API_KEY = os.getenv("JINA_API_KEY", "jina_55408fca253540cb892655bd007f9c264YWeOA2w-l43dEC15Pz_hVf9n2Ue")

# Model configuration
LLM_MODEL = "qwen-plus-2025-12-01"  # Alibaba Cloud Qwen model
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 4000

# Search configuration
MAX_SEARCH_RESULTS_PER_QUERY = 5
MAX_SEARCH_QUERIES_PER_AGENT = 5

# Debate configuration
MAX_DEBATE_ROUNDS = 3
EVIDENCE_POOL_MAX_SIZE = 100

# Priority threshold
PRIORITY_THRESHOLD = 0.05  # Priority differences smaller than this value are considered equal

# Directory configuration
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Logging configuration
LOG_LEVEL = "INFO"
