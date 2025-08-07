#!/usr/bin/env python3

"""
Quick Training Script for Q&A Vector Database
Run this to train your model with question-answer pairs
"""

from training.qa_trainer import qa_trainer

def main():
    print("🎯 Q&A Vector Database Training")
    print("=" * 50)
    
    print("\\nChoose training method:")
    print("1. Enter Q&A pairs manually")
    print("2. Load from file (JSON/CSV)")
    print("3. Use sample data (for testing)")
    
    choice = input("\\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Manual input
        qa_trainer.complete_training_pipeline(method="input")
        
    elif choice == "2":
        # File input
        qa_trainer.complete_training_pipeline(method="file")
        
    elif choice == "3":
        # Sample data for testing
        print("\\n📝 Using sample Q&A data for testing...")
        
        sample_qa = [
            {
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                "category": "AI",
                "difficulty": "beginner"
            },
            {
                "question": "How does deep learning work?",
                "answer": "Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data. Each layer processes the data and passes it to the next layer, allowing the network to learn hierarchical representations.",
                "category": "AI",
                "difficulty": "intermediate"
            },
            {
                "question": "What is natural language processing?",
                "answer": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. It combines computational linguistics with machine learning to process text and speech.",
                "category": "NLP",
                "difficulty": "intermediate"
            },
            {
                "question": "What are the applications of computer vision?",
                "answer": "Computer vision applications include facial recognition, medical image analysis, autonomous vehicles, object detection, image classification, and augmented reality systems.",
                "category": "Computer Vision",
                "difficulty": "intermediate"
            },
            {
                "question": "How do you evaluate machine learning models?",
                "answer": "ML models are evaluated using metrics like accuracy, precision, recall, F1-score, and AUC-ROC. Cross-validation and train/test splits help assess performance on unseen data.",
                "category": "ML Evaluation",
                "difficulty": "advanced"
            }
        ]
        
        # Load model and process data
        qa_trainer.load_embedding_model()
        qa_trainer.add_qa_pairs_manually(sample_qa)
        qa_trainer.generate_embeddings()
        qa_trainer.create_pinecone_index()
        qa_trainer.upload_to_pinecone()
        qa_trainer.save_qa_dataset("sample_qa_dataset.json")
        
        # Test the system
        print("\\n🧪 Testing trained system:")
        qa_trainer.test_retrieval("What is AI?")
        qa_trainer.test_retrieval("How to evaluate models?")
        
        print("\\n🎉 Sample training completed!")
        
    else:
        print("❌ Invalid choice!")
        return
    
    print("\\n✅ Training completed! Your Q&A vector database is ready.")
    print("\\n🔍 Next steps:")
    print("1. Test queries using the query system")
    print("2. Add more Q&A pairs to improve coverage")
    print("3. Fine-tune the system for your specific domain")

if __name__ == "__main__":
    main()
