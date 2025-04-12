import os

from embeddings_lookup import fuse_search_results, get_bm25_results_docs, get_dense_emds_docs, load_bm25_index, load_dense_embeddings
from my_chunking_sys import get_rec_split_chunks, get_sementic_embedding_chunks
from my_data_loader import load_markdowns, load_texts
from hyper_parameters import normal_chunking_param, sementic_chunking_param, bm25_param, dense_embedding_param, reciprocal_rank_fusion_param

#this will load the notes i generated from LLM, although they are in markdown, the format is not what i wanted
#the load_markdown function is not really what i wanted, it just converts the markdownt to html then reads it, which is not ideal since we need markdown formats
#i will work on this later for now just test with these files for retrieval
file_paths = [f for f in os.listdir("doc_mds") if f.endswith(".md")]
file_paths = [os.path.join("doc_mds", f) for f in file_paths]
docs = load_markdowns(file_paths)


#these are the notes from dr iftikhar, they were initially in pdf, but i converted them to text files
#test your resutls seprartely on these files
# file_paths = [f for f in os.listdir("doc2_text") if f.endswith(".txt")]
# file_paths = [os.path.join("doc2_text", f) for f in file_paths]
# docs = load_texts(file_paths)


#these are txt files of some pak history books in online
#test them seperately too
# file_paths = [f for f in os.listdir("docs_txt") if f.endswith(".txt")]
# file_paths = [os.path.join("docs_txt", f) for f in file_paths]
# docs = load_markdowns(file_paths)


#chunks = get_sementic_embedding_chunks(docs, **sementic_chunking_param)
chunks = get_rec_split_chunks(docs, **normal_chunking_param)

texts = [chunk.page_content for chunk in chunks]
metadata = [chunk.metadata for chunk in chunks]

dense_emd_db, emb_model = load_dense_embeddings(chunks, **dense_embedding_param)
bm25_index = load_bm25_index(chunks)

question = "14 points of Jinnah?"

dense_docs = get_dense_emds_docs(dense_emd_db, emb_model, question, k=5)
bm25_docs = get_bm25_results_docs(bm25_index, question, texts, metadata, k=5)

reciprocal_rank_docs = fuse_search_results(dense_emd_db, emb_model, bm25_index, question, texts, metadata, k=5)

# for doc, score in reciprocal_rank_docs:
#     print(f"Document: {doc.page_content}\nScore: {score}\n")

#format retrivals and use it in prompt
def format_response(doc):
    return f"Page {doc.metadata.get('page', 'Unknown')}: {doc.page_content.strip()}"

retrievals = []
for doc, score in reciprocal_rank_docs:
    retrievals.append(format_response(doc))

#make a prompt for llm using retrievals and question
prompt = f"""
You are an AI assistant tasked with answering questions based on retrieved knowledge.

### **Retrieved Information**:
1. {retrievals[0]}

2. {retrievals[1]}

3. {retrievals[2]}

### **Question**:
{question}

### **Instructions**:
- Integrate the key points from all retrieved responses into a **cohesive, well-structured answer**.
- If the responses are **contradictory**, mention the different perspectives.
- If none of the retrieved responses contain relevant information, reply:
  **"I couldn't find a good response to your query in the database."**
"""

print(prompt)