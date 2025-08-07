import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # GitHub Models API
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    GITHUB_ENDPOINT = "https://models.github.ai/inference"
    GITHUB_MODEL = "openai/gpt-4.1"
    
    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    PINECONE_INDEX_NAME = "query-retrieval-index"
    EMBEDDING_DIMENSION = 384  # For sentence-transformers/all-MiniLM-L6-v2
    
    # PostgreSQL Configuration
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "query_retrieval_db")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    
    # Database URL for SQLAlchemy
    DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    
    # Model Configuration
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_SEQUENCE_LENGTH = 512
    BATCH_SIZE = 32
    TOP_K_RESULTS = 10
    
    # Training Configuration
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 32
    
    # Evaluation Metrics
    EVALUATION_METRICS = ["accuracy", "precision", "recall", "f1", "ndcg"]
    
    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
