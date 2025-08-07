# LLM-Powered Query Retrieval System

## Setup Commands

### 1. Install Dependencies
```bash
pip install pinecone-client psycopg2-binary sentence-transformers langchain pandas numpy scikit-learn datasets transformers torch sqlalchemy python-dotenv
```

### 2. Database Setup Commands

#### PostgreSQL Setup
```sql
-- Create database
CREATE DATABASE query_retrieval_db;

-- Create tables
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255),
    content TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE queries (
    id SERIAL PRIMARY KEY,
    query_text TEXT,
    user_id VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE query_results (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES queries(id),
    document_id INTEGER REFERENCES documents(id),
    relevance_score FLOAT,
    rank_position INTEGER
);
```

#### Pinecone Setup Commands
```python
# Run these in Python to set up Pinecone index
import pinecone

# Initialize Pinecone
pinecone.init(api_key="your_api_key", environment="your_environment")

# Create index
pinecone.create_index(
    name="query-retrieval-index",
    dimension=384,  # For sentence-transformers/all-MiniLM-L6-v2
    metric="cosine"
)
```

### 3. Running the System
```bash
# Train/Fine-tune model
python train_model.py

# Process and index documents
python data_processor.py

# Start the query system
python query_system.py

# Run evaluation
python evaluate_system.py
```

### 4. File Structure
```
HACKRX 6.0/
├── .env                    # Environment variables
├── .gitignore             # Git ignore file
├── requirements.txt       # Python dependencies
├── config.py              # Configuration settings
├── database/
│   ├── postgres_setup.py  # PostgreSQL connection and setup
│   └── pinecone_setup.py  # Pinecone configuration
├── models/
│   ├── embedding_model.py # Embedding model wrapper
│   └── retrieval_model.py # Custom retrieval model
├── data/
│   ├── data_processor.py  # Data preprocessing
│   └── dataset_loader.py  # Dataset loading utilities
├── training/
│   ├── train_model.py     # Model training script
│   └── fine_tune.py       # Fine-tuning utilities
├── evaluation/
│   ├── evaluate_system.py # Evaluation metrics
│   └── metrics.py         # Custom metrics
└── api/
    ├── query_system.py    # Main query API
    └── retrieval_api.py   # REST API endpoints
```
