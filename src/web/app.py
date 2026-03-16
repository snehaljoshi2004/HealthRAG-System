# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_from_directory
from src.retrieval.retriever import HybridRetriever
from src.evaluation.evaluate_rag import RAGEvaluator
from ingestion import HealthcareDocumentIngestor
import json
import markdown
import os
import uuid
from werkzeug.utils import secure_filename
import tempfile
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-this-in-production'  # Change this for production
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'md', 'txt', 'html'}

# Initialize RAG components
print("Loading RAG system...")
retriever = HybridRetriever()
evaluator = RAGEvaluator(retriever)
ingestor = HealthcareDocumentIngestor()
print("✅ RAG system ready!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if files were uploaded
        if 'files[]' not in request.files:
            flash('No files selected', 'error')
            return redirect(request.url)
        
        files = request.files.getlist('files[]')
        uploaded_files = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Add timestamp to avoid duplicates
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                uploaded_files.append(filepath)
        
        if uploaded_files:
            # Process and ingest documents
            flash(f'Uploaded {len(uploaded_files)} files. Processing...', 'info')
            
            all_chunks = []
            for filepath in uploaded_files:
                try:
                    if filepath.endswith('.pdf'):
                        chunks = ingestor.load_pdf(filepath)
                    elif filepath.endswith('.md'):
                        chunks = ingestor.load_markdown(filepath)
                    elif filepath.endswith(('.txt', '.html')):
                        chunks = ingestor.load_text(filepath)
                    
                    all_chunks.extend(chunks)
                    
                    # Clean up temp file
                    try:
                        os.remove(filepath)
                    except:
                        pass
                        
                except Exception as e:
                    flash(f'Error processing {os.path.basename(filepath)}: {str(e)}', 'error')
            
            if all_chunks:
                # Ingest to vectorstore
                ingestor.ingest_documents(all_chunks)
                
                # Store info in session
                session['last_upload'] = {
                    'count': len(uploaded_files),
                    'chunks': len(all_chunks)
                }
                
                flash(f'✅ Successfully processed {len(uploaded_files)} files into {len(all_chunks)} chunks!', 'success')
            else:
                flash('No content could be extracted from uploaded files', 'warning')
        
        return redirect(url_for('index'))
    
    return render_template('upload.html')

@app.route('/query', methods=['POST'])
def query():
    question = request.form.get('question', '')
    k = int(request.form.get('k', 5))
    
    # Get results
    results = retriever.retrieve(question, k=k)
    
    # Format for display
    formatted_results = []
    for doc, score in results:
        # Determine score class
        if score > 2.0:
            score_class = "score-high"
        elif score > 1.0:
            score_class = "score-medium"
        else:
            score_class = "score-low"
        
        # Check if document is from upload
        source = doc.metadata.get('source_file', 'Unknown')
        is_uploaded = doc.metadata.get('source_type') == 'upload' or 'upload' in str(doc.metadata)
            
        formatted_results.append({
            'score': f"{score:.4f}",
            'score_class': score_class,
            'source': source,
            'source_type': doc.metadata.get('source_type', 'unknown'),
            'is_uploaded': is_uploaded,
            'content': doc.page_content,
            'preview': doc.page_content[:300] + '...' if len(doc.page_content) > 300 else doc.page_content
        })
    
    return render_template('results.html', 
                         question=question, 
                         results=formatted_results,
                         result_count=len(formatted_results))

@app.route('/evaluate', methods=['GET'])
def evaluate():
    # Run quick evaluation
    golden_data = evaluator.load_golden_dataset(sample_size=10)
    df, summary = evaluator.detailed_evaluation(golden_data)
    
    return render_template('evaluation.html', summary=summary)

@app.route('/api/query', methods=['POST'])
def api_query():
    """JSON API endpoint"""
    data = request.get_json()
    question = data.get('question', '')
    k = data.get('k', 5)
    
    results = retriever.retrieve(question, k=k)
    
    return jsonify([{
        'score': score,
        'source': doc.metadata.get('source_file', 'Unknown'),
        'source_type': doc.metadata.get('source_type', 'unknown'),
        'content': doc.page_content[:500]  # Limit content length
    } for doc, score in results])

@app.route('/clear_uploads', methods=['POST'])
def clear_uploads():
    """Clear uploaded documents from session"""
    session.pop('last_upload', None)
    flash('Upload history cleared', 'info')
    return redirect(url_for('index'))

@app.route('/document/<path:filename>')
def get_document(filename):
    """Serve uploaded documents (if needed)"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)