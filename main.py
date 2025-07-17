# Simple AI Document Retrieval Bot with Gemini
# A complete RAG (Retrieval Augmented Generation) system

import os
import json
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from pathlib import Path
import PyPDF2
import docx
from flask import Flask, request, jsonify, render_template_string

# Configuration
@dataclass
class Config:
    GEMINI_API_KEY: str = "AIzaSyDxuM1K2z4NTcMxAqXBrBLECRg0pSEFfw8"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 3

config = Config()

# Document processor
class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_text_file(self, file_path: str) -> str:
        """Load text from .txt file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def load_pdf_file(self, file_path: str) -> str:
        """Load text from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def load_docx_file(self, file_path: str) -> str:
        """Load text from Word document"""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def load_document(self, file_path: str) -> str:
        """Load document based on file extension"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            return self.load_pdf_file(str(file_path))
        elif file_path.suffix.lower() == '.docx':
            return self.load_docx_file(str(file_path))
        elif file_path.suffix.lower() == '.txt':
            return self.load_text_file(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    def chunk_text(self, text: str, source: str = "") -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'source': source,
                'chunk_id': i // (self.chunk_size - self.chunk_overlap),
                'start_index': i,
                'end_index': min(i + self.chunk_size, len(words)),
                'word_count': len(chunk_words)
            })
        
        return chunks

# Vector database using FAISS
class VectorDatabase:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
        self.embeddings = []
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector database"""
        texts = [doc['text'] for doc in documents]
        embeddings = self.model.encode(texts)
        
        if self.index is None:
            # Initialize FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
        
        # Store documents and embeddings
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Return results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                doc['rank'] = i + 1
                results.append(doc)
        
        return results

# AI Generator using Gemini
class GeminiGenerator:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/gemini-1.5-flash')
    
    def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using Gemini with retrieved context"""
        if not context_chunks:
            return "I don't have enough information to answer that question."
        
        # Prepare context
        context = "\n\n".join([
            f"Document {i+1} (Source: {chunk.get('source', 'Unknown')}):\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, concise answer based on the context
- If you reference specific information, mention which document it came from
- If the answer is not in the provided context, clearly state that
- Be helpful and accurate

Answer:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Main RAG System
class RAGSystem:
    def __init__(self, config: Config):
        self.config = config
        self.processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.vector_db = VectorDatabase(config.EMBEDDING_MODEL)
        self.generator = GeminiGenerator(config.GEMINI_API_KEY)
    
    def add_document(self, file_path: str):
        """Add a document to the system"""
        try:
            # Load document
            text = self.processor.load_document(file_path)
            
            # Chunk document
            chunks = self.processor.chunk_text(text, source=os.path.basename(file_path))
            
            # Add to vector database
            self.vector_db.add_documents(chunks)
            
            return f"Successfully added {len(chunks)} chunks from {file_path}"
        except Exception as e:
            return f"Error processing document: {str(e)}"
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer"""
        try:
            # Retrieve relevant chunks
            relevant_chunks = self.vector_db.search(question, k=self.config.TOP_K)
            
            # Generate answer
            answer = self.generator.generate_answer(question, relevant_chunks)
            
            return {
                'question': question,
                'answer': answer,
                'sources': [
                    {
                        'source': chunk.get('source', 'Unknown'),
                        'score': chunk.get('score', 0),
                        'text_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                    }
                    for chunk in relevant_chunks
                ]
            }
        except Exception as e:
            return {
                'question': question,
                'answer': f"Error processing question: {str(e)}",
                'sources': []
            }

# Flask Web Application
app = Flask(__name__)
rag_system = RAGSystem(config)

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>AI Document Retrieval Bot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .upload-section { margin-bottom: 30px; padding: 20px; background: #f9f9f9; border-radius: 5px; }
        .chat-section { margin-bottom: 30px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user-message { background: #e3f2fd; text-align: right; }
        .bot-message { background: #f3e5f5; }
        .sources { margin-top: 10px; padding: 10px; background: #fff3e0; border-radius: 5px; }
        .source-item { margin: 5px 0; font-size: 12px; color: #666; }
        input[type="text"] { width: 70%; padding: 10px; margin: 5px; }
        button { padding: 10px 20px; margin: 5px; background: #2196F3; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #1976D2; }
        .status { margin: 10px 0; padding: 10px; background: #e8f5e8; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Document Retrieval Bot</h1>
        
        <div class="upload-section">
            <h3>📁 Upload Documents</h3>
            <input type="file" id="fileInput" accept=".txt,.pdf,.docx" multiple>
            <button onclick="uploadDocuments()">Upload</button>
            <div id="uploadStatus" class="status" style="display: none;"></div>
        </div>
        
        <div class="chat-section">
            <h3>💬 Ask Questions</h3>
            <div id="chatHistory" style="height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 10px;"></div>
            <input type="text" id="questionInput" placeholder="Type your question here..." onkeypress="handleKeyPress(event)">
            <button onclick="askQuestion()">Ask</button>
        </div>
    </div>

    <script>
        function uploadDocuments() {
            // Note: This is a simplified frontend. In a real implementation, 
            // you would need to handle file uploads properly
            document.getElementById('uploadStatus').style.display = 'block';
            document.getElementById('uploadStatus').innerHTML = 'File upload functionality would be implemented here. For now, place your documents in the documents/ folder and restart the application.';
        }
        
        function askQuestion() {
            const question = document.getElementById('questionInput').value;
            if (!question.trim()) return;
            
            // Add user message
            const chatHistory = document.getElementById('chatHistory');
            chatHistory.innerHTML += `<div class="message user-message"><strong>You:</strong> ${question}</div>`;
            
            // Clear input
            document.getElementById('questionInput').value = '';
            
            // Send question to backend
            fetch('/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: question})
            })
            .then(response => response.json())
            .then(data => {
                let sourcesHtml = '';
                if (data.sources && data.sources.length > 0) {
                    sourcesHtml = '<div class="sources"><strong>Sources:</strong><br>';
                    data.sources.forEach((source, index) => {
                        sourcesHtml += `<div class="source-item">${index + 1}. ${source.source} (Score: ${source.score.toFixed(3)})</div>`;
                    });
                    sourcesHtml += '</div>';
                }
                
                chatHistory.innerHTML += `<div class="message bot-message"><strong>Bot:</strong> ${data.answer}${sourcesHtml}</div>`;
                chatHistory.scrollTop = chatHistory.scrollHeight;
            })
            .catch(error => {
                chatHistory.innerHTML += `<div class="message bot-message"><strong>Bot:</strong> Error: ${error}</div>`;
            });
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                askQuestion();
            }
        }
    </script>
</body>
</html>
    ''')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    
    result = rag_system.ask_question(question)
    return jsonify(result)

# Command Line Interface
def main():
    print("🤖 AI Document Retrieval Bot with Gemini")
    print("=" * 50)
    
    # Check if API key is set
    if config.GEMINI_API_KEY == "your-gemini-api-key-here":
        print("⚠️  Please set your Gemini API key in the config!")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Load documents from documents folder
    documents_folder = Path("documents")
    if documents_folder.exists():
        print(f"📁 Loading documents from {documents_folder}...")
        for file_path in documents_folder.glob("*"):
            if file_path.suffix.lower() in ['.txt', '.pdf', '.docx']:
                result = rag_system.add_document(str(file_path))
                print(f"   {result}")
    else:
        print("📁 Create a 'documents' folder and add your files there")
        documents_folder.mkdir(exist_ok=True)
    
    print("\n" + "=" * 50)
    print("Choose an option:")
    print("1. Start web interface")
    print("2. Command line chat")
    print("3. Add more documents")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        print("\n🌐 Starting web interface...")
        print("Open your browser and go to: http://localhost:5000")
        app.run(debug=True, port=5000)
    
    elif choice == "2":
        print("\n💬 Starting command line chat...")
        print("Type 'quit' to exit")
        
        while True:
            question = input("\n❓ Ask a question: ")
            if question.lower() == 'quit':
                break
            
            result = rag_system.ask_question(question)
            print(f"\n🤖 Answer: {result['answer']}")
            
            if result['sources']:
                print("\n📚 Sources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"   {i}. {source['source']} (Score: {source['score']:.3f})")
    
    elif choice == "3":
        file_path = input("Enter the path to your document: ")
        result = rag_system.add_document(file_path)
        print(result)

if __name__ == "__main__":
    main()


# Installation Requirements (requirements.txt):
"""
flask==2.3.3
sentence-transformers==2.2.2
faiss-cpu==1.7.4
google-generativeai==0.3.2
PyPDF2==3.0.1
python-docx==0.8.11
numpy==1.24.3
pathlib
"""

# Setup Instructions:
"""
1. Install Python 3.8+
2. Install requirements: pip install -r requirements.txt
3. Get Gemini API key from: https://makersuite.google.com/app/apikey
4. Replace 'your-gemini-api-key-here' with your actual API key
5. Create a 'documents' folder and add your PDF/Word/Text files
6. Run: python main.py
"""