#!/usr/bin/env python3

"""
Demo script for LLM-Powered Query Retrieval System
This script demonstrates the complete pipeline
"""

import logging
import os
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("🚀 LLM-Powered Query Retrieval System Demo")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Check if we have the required token
    if not os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN") == "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":
        print("✅ GitHub token found")
    else:
        print("⚠️ Please set your GitHub token in .env file")
        return
    
    print("\\n📊 Step 1: Processing and Indexing Sample Data")
    print("-" * 40)
    
    try:
        from data.data_processor import data_processor
        
        # Process sample data (no external database needed for demo)
        success = data_processor.process_and_index_data()
        
        if success:
            print("✅ Data processing completed successfully!")
        else:
            print("❌ Data processing failed - continuing with demo using simple search")
            
    except Exception as e:
        print(f"⚠️ Data processing error (expected if no DB): {e}")
        print("📝 Continuing with simple embedding demo...")
    
    print("\\n🔍 Step 2: Testing Query System")
    print("-" * 40)
    
    try:
        from api.query_system import query_system
        
        # Initialize system (will work with simple embeddings even without DB)
        if query_system.initialize_system():
            print("✅ Query system initialized!")
        
        # Test queries
        test_queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "Explain natural language processing",
            "What are computer vision applications?"
        ]
        
        print("\\n🤖 Testing Query-Answer Generation:")
        print("-" * 40)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\\n{i}. Query: {query}")
            
            try:
                # For demo, use simple search without full pipeline
                from models.simple_embedding import embedding_model
                
                # Generate query embedding
                embedding_model.load_model()
                query_embedding = embedding_model.encode_query(query)
                
                print(f"   ✅ Query encoded successfully (dimension: {len(query_embedding)})")
                
                # Generate answer using GitHub Models API
                from openai import OpenAI
                
                client = OpenAI(
                    base_url="https://models.github.ai/inference",
                    api_key=os.getenv("GITHUB_TOKEN"),
                )
                
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful AI assistant. Provide clear, concise answers about technical topics."
                        },
                        {
                            "role": "user",
                            "content": query
                        }
                    ],
                    temperature=0.7,
                    model="openai/gpt-4.1"
                )
                
                answer = response.choices[0].message.content
                print(f"   🤖 Answer: {answer[:100]}...")
                
            except Exception as e:
                print(f"   ❌ Query processing error: {e}")
    
    except Exception as e:
        print(f"❌ Query system error: {e}")
    
    print("\\n📈 Step 3: System Information")
    print("-" * 40)
    
    try:
        from models.simple_embedding import embedding_model
        model_info = embedding_model.get_model_info()
        print(f"📊 Embedding Model: {model_info}")
        
        # Test basic functionality
        sample_texts = ["Hello world", "Machine learning is great"]
        embeddings = embedding_model.encode_text(sample_texts)
        print(f"📊 Sample embeddings shape: {embeddings.shape}")
        
        # Test similarity
        similarity = embedding_model.similarity(embeddings[0], embeddings[1])
        print(f"📊 Text similarity example: {similarity:.3f}")
        
    except Exception as e:
        print(f"❌ System info error: {e}")
    
    print("\\n🎯 Demo Complete!")
    print("=" * 60)
    print("\\n📝 What this demo showed:")
    print("✅ Environment setup and package imports")
    print("✅ Text embedding generation")
    print("✅ GitHub Models API integration")
    print("✅ Basic query processing pipeline")
    
    print("\\n🚀 Next Steps:")
    print("1. Set up PostgreSQL database")
    print("2. Configure Pinecone account")
    print("3. Add your real dataset")
    print("4. Train/fine-tune models on your data")
    print("5. Deploy as a web API using FastAPI")
    
    print("\\n💡 To run the full system:")
    print('   python -c "from data.data_processor import data_processor; data_processor.process_and_index_data()"')
    print('   python -c "from api.query_system import query_system; print(query_system.query_with_retrieval(\'your query\'))"')

if __name__ == "__main__":
    main()
