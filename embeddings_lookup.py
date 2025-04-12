from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from rank_bm25 import BM25Okapi
from langchain.schema import Document
from hyper_parameters import dense_embedding_param, bm25_param
from hyper_parameters import reciprocal_rank_fusion_param



def load_dense_embeddings(docs, model_name="BAAI/bge-small-en", distance_strategy=DistanceStrategy.COSINE, index_path=None):

    # Initialize embeddings
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    
    # Create a FAISS vector store
    res_emds = FAISS.from_documents(
            docs, embedding_model, distance_strategy=distance_strategy
        )
    res_emds.save_local(index_path)

    return res_emds, embedding_model

def load_bm25_index(docs):
    """Create a BM25 index from the documents."""
    texts = [doc.page_content for doc in docs]
    tokenized_texts = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts, **bm25_param)
    return bm25

def get_dense_emds_docs(emb_db, emb_model, query, k=5):

    query_embedding = emb_model.embed_query(query)
    results_embedding = emb_db.similarity_search_with_score_by_vector(query_embedding, k=k)
    results_embedding = sorted(results_embedding, key=lambda x: x[1], reverse=True)

    return results_embedding

def get_bm25_results_docs(bm25, query, texts, metadata, k=5):
    # Get BM25 scores for all documents and sort to get top-k results
    results_bm25 = [(idx, bm25.get_scores(query.split())[idx]) for idx in range(len(texts))]
    results_bm25 = sorted(results_bm25, key=lambda x: x[1], reverse=True)[:k]  # Keep only top-k results
    # Convert BM25 results to (Document, score) format
    results_bm25_docs = [(Document(page_content=texts[idx], metadata=metadata[idx]), score) for idx, score in results_bm25]

    return results_bm25_docs

def reciprocal_rank_fusion(results_lists, k=60):
    # Create a dictionary to store fusion scores for each document
    fusion_scores = {}
    
    # Process each result list
    for results in results_lists:
        # Get the rank of each document in this result list
        for rank, (doc, _) in enumerate(results):
            # Use document content as key (assuming Document objects have unique content)
            doc_key = doc.page_content
            
            # Calculate reciprocal rank with k constant
            # rank + 1 because rank is 0-indexed but RRF uses 1-indexed ranks
            rrf_score = 1 / (k + rank + 1)
            
            # Add to fusion scores, creating entry if it doesn't exist
            if doc_key not in fusion_scores:
                fusion_scores[doc_key] = {"doc": doc, "score": 0}
            
            # Accumulate scores
            fusion_scores[doc_key]["score"] += rrf_score
    
    # Convert dictionary to list of (Document, score) tuples
    fused_results = [(item["doc"], item["score"]) for item in fusion_scores.values()]
    
    # Sort by fusion score in descending order
    fused_results = sorted(fused_results, key=lambda x: x[1], reverse=True)
    
    return fused_results

def fuse_search_results(emb_db, emb_model, bm25, query, texts, metadata, k=5):
    # Get results from embedding-based search
    dense_results = get_dense_emds_docs(emb_db, emb_model, query, k=k)   
    # Get results from BM25 search
    bm25_results = get_bm25_results_docs(bm25, query, texts, metadata, k=k) 
    # Fuse results using RRF
    fused_results = reciprocal_rank_fusion([dense_results, bm25_results], **reciprocal_rank_fusion_param)
    
    # Return top-k fused results
    return fused_results[:k]


#re-ranking for later
