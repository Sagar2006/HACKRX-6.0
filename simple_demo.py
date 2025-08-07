#!/usr/bin/env python3

"""
Simple Demo - LLM Query System without Database
This works with just GitHub Models API and simple embeddings
"""

import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load environment
load_dotenv()

def main():
    print("🚀 Simple LLM Query System Demo")
    print("=" * 50)
    
    # Initialize embedding model
    from models.simple_embedding import embedding_model
    embedding_model.load_model()
    
    # Sample knowledge base
    knowledge_base = {
        "Machine Learning": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
        "Deep Learning": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data, particularly useful for image and text processing.",
        "Natural Language Processing": "NLP combines computational linguistics with machine learning to help computers understand, interpret, and generate human language in a valuable way.",
        "Computer Vision": "Computer vision enables machines to interpret and understand visual information from the world, including image recognition, object detection, and scene understanding.",
        "Data Science": "Data science involves collecting, cleaning, analyzing, and interpreting large amounts of data to extract meaningful insights and support decision-making."
    }
    
    # Create embeddings for knowledge base
    print("📊 Creating knowledge base embeddings...")
    kb_texts = list(knowledge_base.values())
    kb_titles = list(knowledge_base.keys())
    kb_embeddings = embedding_model.encode_documents(kb_texts)
    
    # Initialize GitHub Models API
    client = OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=os.getenv("GITHUB_TOKEN"),
    )
    
    def search_and_answer(query):
        print(f"\\n🔍 Processing query: '{query}'")
        
        # 1. Encode query
        query_embedding = embedding_model.encode_query(query)
        
        # 2. Find most similar documents
        similarities = []
        for i, kb_emb in enumerate(kb_embeddings):
            similarity = embedding_model.similarity(query_embedding, kb_emb)
            similarities.append((similarity, i, kb_titles[i], kb_texts[i]))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        top_results = similarities[:3]
        
        print(f"   📋 Top {len(top_results)} relevant documents:")
        for rank, (score, idx, title, text) in enumerate(top_results, 1):
            print(f"     {rank}. {title} (similarity: {score:.3f})")
        
        # 3. Generate answer using top results as context
        context = "\\n\\n".join([f"{title}: {text}" for _, _, title, text in top_results])
        
        system_prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

Context:
{context}

Instructions:
- Use the context to provide accurate information
- Be concise but comprehensive
- If the context doesn't fully answer the question, mention what information is available"""

        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                model="openai/gpt-4.1"
            )
            
            answer = response.choices[0].message.content
            print(f"   🤖 Answer: {answer}")
            
        except Exception as e:
            print(f"   ❌ Error generating answer: {e}")
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How does deep learning differ from traditional ML?",
        "What can computer vision do?",
        "Explain data science process",
        "What is the difference between AI and machine learning?"
    ]
    
    print("\\n🎯 Testing Query System:")
    print("-" * 40)
    
    for query in test_queries:
        search_and_answer(query)
    
    print("\\n✅ Demo Complete!")
    print("\\n📊 System Stats:")
    print(f"   - Knowledge base size: {len(knowledge_base)} documents")
    print(f"   - Embedding dimension: {embedding_model.dimension}")
    print(f"   - Model type: {embedding_model.get_model_info()['type']}")
    
    print("\\n🚀 This demo shows:")
    print("   ✅ Text embedding and similarity search")
    print("   ✅ Retrieval-Augmented Generation (RAG)")
    print("   ✅ GitHub Models API integration")
    print("   ✅ Complete query-answer pipeline")

if __name__ == "__main__":
    main()
