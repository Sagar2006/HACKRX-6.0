import psycopg2
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import Config
import logging

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255))
    content = Column(Text)
    doc_metadata = Column(JSON)  # Changed from 'metadata' to 'doc_metadata'
    created_at = Column(TIMESTAMP)

class Query(Base):
    __tablename__ = 'queries'
    
    id = Column(Integer, primary_key=True)
    query_text = Column(Text)
    user_id = Column(String(100))
    timestamp = Column(TIMESTAMP)

class QueryResult(Base):
    __tablename__ = 'query_results'
    
    id = Column(Integer, primary_key=True)
    query_id = Column(Integer)
    document_id = Column(Integer)
    relevance_score = Column(Float)
    rank_position = Column(Integer)

class PostgreSQLManager:
    def __init__(self):
        self.engine = create_engine(Config.DATABASE_URL)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all tables in the database"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logging.info("Database tables created successfully")
        except Exception as e:
            logging.error(f"Error creating tables: {e}")
            
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def insert_document(self, title, content, metadata=None):
        """Insert a document into the database"""
        session = self.get_session()
        try:
            doc = Document(title=title, content=content, doc_metadata=metadata)
            session.add(doc)
            session.commit()
            doc_id = doc.id
            session.close()
            return doc_id
        except Exception as e:
            session.rollback()
            session.close()
            logging.error(f"Error inserting document: {e}")
            return None
    
    def get_documents(self, limit=None):
        """Retrieve documents from database"""
        session = self.get_session()
        try:
            query = session.query(Document)
            if limit:
                query = query.limit(limit)
            documents = query.all()
            session.close()
            return documents
        except Exception as e:
            session.close()
            logging.error(f"Error retrieving documents: {e}")
            return []
    
    def log_query(self, query_text, user_id="anonymous"):
        """Log a user query"""
        session = self.get_session()
        try:
            query = Query(query_text=query_text, user_id=user_id)
            session.add(query)
            session.commit()
            query_id = query.id
            session.close()
            return query_id
        except Exception as e:
            session.rollback()
            session.close()
            logging.error(f"Error logging query: {e}")
            return None
    
    def log_query_results(self, query_id, results):
        """Log query results with relevance scores"""
        session = self.get_session()
        try:
            for rank, (doc_id, score) in enumerate(results, 1):
                result = QueryResult(
                    query_id=query_id,
                    document_id=doc_id,
                    relevance_score=score,
                    rank_position=rank
                )
                session.add(result)
            session.commit()
            session.close()
            logging.info(f"Logged {len(results)} query results")
        except Exception as e:
            session.rollback()
            session.close()
            logging.error(f"Error logging query results: {e}")

# Initialize database manager
db_manager = PostgreSQLManager()
