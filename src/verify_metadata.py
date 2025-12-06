#!/usr/bin/env python3
"""Verify that vector store documents have correct BibTeX metadata"""
import os
import sys
from pathlib import Path
from langchain_community.vectorstores import FAISS
from Knowledge import get_embedding_model, parse_bibtex_metadata

def verify_vectorstore_metadata():
    """Verify metadata in vector store matches references.bib"""
    
    # Load BibTeX metadata
    bib_path = Path(__file__).parent / "books_pdf" / "references.bib"
    bib_metadata = parse_bibtex_metadata(str(bib_path))
    
    print(f"📚 Loaded {len(bib_metadata)} BibTeX entries from {bib_path}\n")
    
    # Load vector store
    embedding_model = get_embedding_model()
    vs_path = Path(__file__).parent / "vector_store" / "truenorth_kb_vectorstore"
    
    if not vs_path.exists():
        print(f"❌ Vector store not found at {vs_path}")
        print("   Run 'make embed' to build the vector store first.")
        return False
    
    vs = FAISS.load_local(str(vs_path), embedding_model, allow_dangerous_deserialization=True)
    print(f"📦 Loaded vector store with {vs.index.ntotal} documents\n")
    
    # Test queries to get sample documents
    test_queries = [
        'workplace wellbeing',
        'negotiation strategies',
        'principal investigator',
        'PERMA+4',
        'mentorship STEM'
    ]
    
    all_docs_checked = set()
    stats = {
        'total_checked': 0,
        'has_metadata': 0,
        'missing_metadata': 0,
        'missing_from_bib': 0,
        'mismatched_metadata': 0
    }
    
    print("=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    
    for query in test_queries:
        docs = vs.similarity_search(query, k=10)
        
        for doc in docs:
            filename = doc.metadata.get("filename", "Unknown")
            
            # Skip if we've already checked this document
            if filename in all_docs_checked:
                continue
            all_docs_checked.add(filename)
            
            stats['total_checked'] += 1
            
            author = doc.metadata.get("author", None)
            title = doc.metadata.get("title", None)
            year = doc.metadata.get("year", None)
            
            # Check if file has BibTeX entry
            bib_entry = bib_metadata.get(filename, None)
            
            if not bib_entry:
                stats['missing_from_bib'] += 1
                status = "⚠️  MISSING FROM BIBTEX"
            elif not author or not title or author == "Unknown Author" or title == "Unknown Title":
                stats['missing_metadata'] += 1
                status = "❌ MISSING METADATA"
            else:
                # Check if metadata matches BibTeX
                bib_author = bib_entry.get("author", "").strip()
                bib_title = bib_entry.get("title", "").strip()
                bib_year = bib_entry.get("year", "").strip()
                
                # Normalize for comparison (remove BOM, extra spaces)
                doc_author = str(author).replace("\ufeff", "").strip() if author else ""
                doc_title = str(title).replace("\ufeff", "").strip() if title else ""
                doc_year = str(year).strip() if year else ""
                
                # Check if metadata matches (allowing for some variation)
                author_match = bib_author.lower() in doc_author.lower() or doc_author.lower() in bib_author.lower() if bib_author and doc_author else False
                title_match = bib_title.lower() in doc_title.lower() or doc_title.lower() in bib_title.lower() if bib_title and doc_title else False
                year_match = bib_year == doc_year if bib_year and doc_year else True  # Year optional
                
                if author_match and title_match:
                    stats['has_metadata'] += 1
                    status = "✅ HAS METADATA"
                else:
                    stats['mismatched_metadata'] += 1
                    status = "⚠️  MISMATCHED METADATA"
            
            print(f"\n{status}: {filename}")
            print(f"  Author: {author or 'NOT FOUND'}")
            print(f"  Title: {title or 'NOT FOUND'}")
            print(f"  Year: {year or 'NOT FOUND'}")
            
            if bib_entry:
                print(f"  BibTeX Author: {bib_entry.get('author', 'N/A')}")
                print(f"  BibTeX Title: {bib_entry.get('title', 'N/A')}")
                print(f"  BibTeX Year: {bib_entry.get('year', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total documents checked: {stats['total_checked']}")
    print(f"✅ Has correct metadata: {stats['has_metadata']} ({stats['has_metadata']/max(stats['total_checked'],1)*100:.1f}%)")
    print(f"❌ Missing metadata: {stats['missing_metadata']} ({stats['missing_metadata']/max(stats['total_checked'],1)*100:.1f}%)")
    print(f"⚠️  Missing from BibTeX: {stats['missing_from_bib']} ({stats['missing_from_bib']/max(stats['total_checked'],1)*100:.1f}%)")
    print(f"⚠️  Mismatched metadata: {stats['mismatched_metadata']} ({stats['mismatched_metadata']/max(stats['total_checked'],1)*100:.1f}%)")
    
    # Overall status
    success_rate = stats['has_metadata'] / max(stats['total_checked'], 1) * 100
    if success_rate >= 90:
        print(f"\n✅ SUCCESS: {success_rate:.1f}% of documents have correct metadata")
        return True
    elif success_rate >= 70:
        print(f"\n⚠️  WARNING: {success_rate:.1f}% of documents have correct metadata")
        print("   Consider rebuilding the vector store to update metadata.")
        return False
    else:
        print(f"\n❌ FAILURE: Only {success_rate:.1f}% of documents have correct metadata")
        print("   Rebuild the vector store with 'make embed' to fix this.")
        return False

if __name__ == "__main__":
    success = verify_vectorstore_metadata()
    sys.exit(0 if success else 1)

