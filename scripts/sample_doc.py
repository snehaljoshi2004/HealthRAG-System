# sample_doc.py
import json
import os
import re

def clean_filename(filename):
    """Clean filename to be safe for Windows"""
    # Remove or replace invalid characters
    # Windows invalid chars: \ / : * ? " < > |
    filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
    # Replace newlines and carriage returns
    filename = filename.replace('\n', '_').replace('\r', '_')
    # Replace multiple underscores with single
    filename = re.sub(r'_+', '_', filename)
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    # Remove trailing spaces and dots
    filename = filename.strip('. ')
    return filename

def create_sample_documents():
    print("Creating sample healthcare documents...")
    
    # Load the clean dataset
    with open('data/evaluation/rag_dataset_clean.json', 'r') as f:
        dataset = json.load(f)
    
    # Create data/raw directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Group contexts by source document
    documents = {}
    for item in dataset:
        source = item['metadata']['source']
        if source not in documents:
            documents[source] = {
                'title': source,
                'pages': {},
                'content': []
            }
        
        # Add context to document
        page = item['metadata']['page']
        if page not in documents[source]['pages']:
            documents[source]['pages'][page] = []
        
        documents[source]['pages'][page].append(item['context'])
        documents[source]['content'].append(item['context'])
    
    # Create markdown files for each source
    created_files = []
    for source, doc_info in documents.items():
        if source == 'Unknown' or not source:
            continue
            
        # Clean the filename
        clean_name = clean_filename(source)
        filepath = f"data/raw/{clean_name}.md"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {doc_info['title']}\n\n")
                
                # Write content by page
                for page_num in sorted(doc_info['pages'].keys()):
                    if page_num and page_num != 'Unknown' and page_num != 'nan':
                        f.write(f"## Page {page_num}\n\n")
                    else:
                        f.write(f"## Section\n\n")
                        
                    for context in doc_info['pages'][page_num]:
                        f.write(f"{context}\n\n")
            
            print(f"   ✅ Created: {filepath}")
            created_files.append(filepath)
            
        except Exception as e:
            print(f"   ❌ Error creating {clean_name}: {str(e)}")
            # Try with an even simpler filename
            simple_name = f"document_{len(created_files)}.md"
            simple_path = f"data/raw/{simple_name}"
            with open(simple_path, 'w', encoding='utf-8') as f:
                f.write(f"# {doc_info['title']}\n\n")
                for context in doc_info['content']:
                    f.write(f"{context}\n\n")
            print(f"   ✅ Created simplified: {simple_path}")
            created_files.append(simple_path)
    
    print(f"\n✅ Created {len(created_files)} sample documents in data/raw/")
    
    # List the created files
    print("\n📁 Files created:")
    for file in created_files:
        print(f"   - {os.path.basename(file)}")
    
    return created_files

if __name__ == "__main__":
    created_files = create_sample_documents()
    
    # Show first few lines of first file
    if created_files:
        print(f"\n📄 Preview of first file ({os.path.basename(created_files[0])}):")
        with open(created_files[0], 'r', encoding='utf-8') as f:
            lines = f.readlines()[:5]
            for line in lines:
                print(f"   {line.rstrip()}")