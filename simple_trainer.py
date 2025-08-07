#!/usr/bin/env python3

"""
Simple Q&A Training System (without sentence-transformers dependencies)
"""

import json
import numpy as np
from typing import List, Dict, Any
from models.simple_embedding import embedding_model
from database.pinecone_setup import pinecone_manager
from config import Config

def simple_qa_training():
    print("🚀 Simple Q&A Vector Database Training")
    print("=" * 60)
    
    # Sample Q&A data (you can replace this with your data)
    print("\\n📝 Would you like to:")
    print("1. Enter your Q&A pairs manually")
    print("2. Use sample data for testing")
    
    choice = input("\\nChoice (1-2): ").strip()
    
    qa_pairs = []
    
    if choice == "1":
        print("\\n📋 Enter your Question-Answer pairs (type 'done' when finished):")
        
        while True:
            print(f"\\n📝 Q&A Pair #{len(qa_pairs) + 1}:")
            question = input("❓ Question: ").strip()
            if question.lower() == 'done':
                break
                
            answer = input("💬 Answer: ").strip()
            if answer.lower() == 'done':
                break
                
            category = input("🏷️ Category (optional): ").strip() or "general"
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "category": category,
                "id": f"qa_{len(qa_pairs) + 1}"
            })
            
            print(f"✅ Added Q&A pair #{len(qa_pairs)}")
    
    else:
        # Use sample data
        qa_pairs = [
            {
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                "category": "AI",
                "id": "qa_1"
            },
            {
                "question": "How does deep learning work?",
                "answer": "Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data. Each layer processes the data and passes it to the next layer.",
                "category": "AI",
                "id": "qa_2"
            },
            {
                "question": "What is natural language processing?",
                "answer": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language using computational linguistics and machine learning.",
                "category": "NLP",
                "id": "qa_3"
            },
            {
                "question": "What are computer vision applications?",
                "answer": "Computer vision applications include facial recognition, medical image analysis, autonomous vehicles, object detection, image classification, and augmented reality systems.",
                "category": "Computer Vision",
                "id": "qa_4"
            },
            {
                "question": "How do you evaluate machine learning models?",
                "answer": "ML models are evaluated using metrics like accuracy, precision, recall, F1-score, and AUC-ROC. Cross-validation and train/test splits help assess performance on unseen data.",
                "category": "ML Evaluation",
                "id": "qa_5"
            }
        ]
        print(f"\\n📊 Using {len(qa_pairs)} sample Q&A pairs")
    
    if not qa_pairs:
        print("❌ No Q&A pairs to process!")
        return
    
    # Step 1: Load embedding model
    print("\\n🔄 Loading embedding model...")
    if not embedding_model.load_model():
        print("❌ Failed to load embedding model")
        return
    
    # Step 2: Generate embeddings
    print("🔄 Generating embeddings...")
    
    # Prepare texts
    questions = [qa['question'] for qa in qa_pairs]
    answers = [qa['answer'] for qa in qa_pairs]
    combined_texts = [f"Question: {qa['question']} Answer: {qa['answer']}" for qa in qa_pairs]
    
    # Generate embeddings
    question_embeddings = embedding_model.encode_documents(questions)
    combined_embeddings = embedding_model.encode_documents(combined_texts)
    
    print(f"✅ Generated embeddings for {len(qa_pairs)} Q&A pairs")
    
    # Step 3: Setup Pinecone
    print("🔄 Setting up Pinecone...")
    Config.PINECONE_INDEX_NAME = "qa-knowledge-base"
    pinecone_manager.index_name = "qa-knowledge-base"
    
    if not pinecone_manager.initialize():
        print("❌ Failed to initialize Pinecone")
        return
    
    print("✅ Pinecone connected!")
    
    # Step 4: Upload to Pinecone
    print("🔄 Uploading to vector database...")
    
    vectors = []
    for i, qa in enumerate(qa_pairs):
        # Question vector
        question_vector = (
            f"q_{qa['id']}",
            question_embeddings[i].tolist(),
            {
                "type": "question",
                "question": qa['question'][:500],
                "answer": qa['answer'][:500],
                "category": qa['category'],
                "qa_id": qa['id']
            }
        )
        vectors.append(question_vector)
        
        # Combined vector for better context
        combined_vector = (
            f"c_{qa['id']}",
            combined_embeddings[i].tolist(),
            {
                "type": "combined",
                "question": qa['question'][:400],
                "answer": qa['answer'][:400],
                "category": qa['category'],
                "qa_id": qa['id']
            }
        )
        vectors.append(combined_vector)
    
    # Upload vectors
    success = pinecone_manager.upsert_vectors(vectors)
    
    if success:
        print(f"✅ Uploaded {len(vectors)} vectors to Pinecone!")
        
        # Get stats
        stats = pinecone_manager.get_index_stats()
        if stats:
            print(f"📊 Index stats: {stats}")
    else:
        print("❌ Failed to upload vectors")
        return
    
    # Step 5: Save dataset
    print("💾 Saving Q&A dataset...")
    with open("qa_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print("✅ Dataset saved to qa_dataset.json")
    
    # Step 6: Test the system
    print("\\n🧪 Testing the trained system:")
    test_question = qa_pairs[0]['question']
    
    # Generate test embedding
    test_embedding = embedding_model.encode_query(test_question)
    
    # Search
    results = pinecone_manager.search_similar(
        query_vector=test_embedding.tolist(),
        top_k=3
    )
    
    print(f"\\n🔍 Test query: '{test_question}'")
    print("📋 Results:")
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        print(f"\\n{i}. Score: {result['score']:.3f}")
        print(f"   Q: {metadata['question']}")
        print(f"   A: {metadata['answer']}")
    
    print("\\n🎉 Training completed successfully!")
    print(f"📊 Processed {len(qa_pairs)} Q&A pairs")
    print("🔍 Your vector database is ready for queries!")
    
    # Additional API keys info
    print("\\n💡 Additional setup (if needed):")
    print("- For better embeddings: OpenAI API key")
    print("- For PostgreSQL storage: Database credentials")
    print("- Current setup uses Pinecone + GitHub Models (working!)")

if __name__ == "__main__":
    simple_qa_training()
