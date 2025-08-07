#!/usr/bin/env python3

"""
Direct training script for insurance Q&A data
"""

import json
from local_trainer import LocalVectorDB
from models.simple_embedding import embedding_model

def train_insurance_qa():
    print("🚀 Training Insurance Q&A Vector Database")
    print("=" * 60)
    
    # Load insurance Q&A data
    with open("insurance_qa_dataset.json", 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    print(f"📊 Loaded {len(qa_pairs)} insurance Q&A pairs")
    
    # Initialize components
    local_db = LocalVectorDB()
    
    # Load embedding model
    print("🔄 Loading embedding model...")
    if not embedding_model.load_model():
        print("❌ Failed to load embedding model")
        return None
    
    # Generate embeddings
    print("🔄 Generating embeddings...")
    
    questions = [qa['question'] for qa in qa_pairs]
    answers = [qa['answer'] for qa in qa_pairs]
    combined_texts = [f"Question: {qa['question']} Answer: {qa['answer']}" for qa in qa_pairs]
    
    question_embeddings = embedding_model.encode_documents(questions)
    combined_embeddings = embedding_model.encode_documents(combined_texts)
    
    print(f"✅ Generated embeddings for {len(qa_pairs)} Q&A pairs")
    
    # Build vector database
    print("🔄 Building vector database...")
    
    vectors = []
    for i, qa in enumerate(qa_pairs):
        # Question vector
        question_vector = (
            f"q_{qa['id']}",
            question_embeddings[i].tolist(),
            {
                "type": "question",
                "question": qa['question'],
                "answer": qa['answer'],
                "category": qa['category'],
                "qa_id": qa['id']
            }
        )
        vectors.append(question_vector)
        
        # Combined vector
        combined_vector = (
            f"c_{qa['id']}",
            combined_embeddings[i].tolist(),
            {
                "type": "combined",
                "question": qa['question'],
                "answer": qa['answer'],
                "category": qa['category'],
                "qa_id": qa['id']
            }
        )
        vectors.append(combined_vector)
    
    local_db.add_vectors(vectors)
    print(f"✅ Added {len(vectors)} vectors to database")
    
    # Save databases
    print("💾 Saving databases...")
    local_db.save("local_vector_db.json")
    
    # Copy the dataset file
    with open("qa_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    print("✅ Databases saved!")
    
    # Test the system
    print("\\n🧪 Testing the trained system:")
    
    test_questions = [
        "What is the grace period for premium payment?",
        "How long is the waiting period for pre-existing diseases?",
        "Does the policy cover maternity expenses?"
    ]
    
    for test_question in test_questions:
        print(f"\\n🔍 Test query: '{test_question}'")
        
        test_embedding = embedding_model.encode_query(test_question)
        results = local_db.search_similar(test_embedding.tolist(), top_k=2)
        
        print("📋 Top results:")
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            print(f"\\n{i}. Score: {result['score']:.3f}")
            print(f"   Q: {metadata['question'][:80]}...")
            print(f"   A: {metadata['answer'][:80]}...")
            print(f"   Category: {metadata['category']}")
    
    # Show stats
    stats = local_db.get_stats()
    print(f"\\n📊 Database Stats: {stats}")
    
    print("\\n🎉 Insurance Q&A training completed successfully!")
    print(f"📊 Processed {len(qa_pairs)} Q&A pairs")
    print("🔍 Your insurance knowledge base is ready!")
    
    return local_db

if __name__ == "__main__":
    train_insurance_qa()
