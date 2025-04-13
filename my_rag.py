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

#format retrivals and use it in prompt
def format_response(doc):
    return f"Page {doc.metadata.get('page', 'Unknown')}: {doc.page_content.strip()}"

qa_pairs = [
    # Simple questions
    ("What was the Khilafat Movement?", "A movement (1919–1924) aimed at protecting the Ottoman Caliphate after World War I."),
    ("Why did Indian Muslims support the Khilafat Movement?", "They revered the Ottoman Caliph as the symbolic leader of the global Muslim community."),
    # ("What treaty threatened the Ottoman Caliphate?", "The Treaty of Sèvres (1920)."),
    # ("What was a key religious concern of Indian Muslims during the Khilafat Movement?", "The potential loss of control over Islamic holy sites in the Hejaz region, including Mecca and Medina."),
    # ("Who were the key leaders of the Khilafat Movement?", "Leaders like Maulana Muhammad Ali Johar."),
    
    # # Intermediate questions
    # ("What were the objectives of the Khilafat Conference in 1918?", "Preservation of the Ottoman Caliphate, protection of its territorial integrity, and mobilization of Indian Muslims."),
    # ("How did the Khilafat Movement utilize non-cooperation tactics?", "It included boycotts of British goods and institutions, aligning with Mahatma Gandhi’s broader non-cooperation strategy."),
    # ("What was the purpose of delegations sent to England (1919–1921)?", "To persuade British authorities to protect the Caliphate and Ottoman territories."),
    # ("Why did the British government reject the demands of the Khilafat Movement?", "The British prioritized geopolitical interests over Indian Muslim concerns."),
    
    # # Complex questions
    # ("What was the Hijrat Movement, and why did it fail?", "A migration effort where Indian Muslims sought refuge in Afghanistan, but poor planning and Afghanistan's unpreparedness led to failure."),
    # ("How did the alliance between Gandhi and the Khilafat Movement influence Hindu-Muslim relations?", "It briefly strengthened unity, but later tensions resurfaced due to differing political objectives."),
    # ("What were the internal challenges faced by the Khilafat Movement?", "Diverse objectives and leadership fragmentation weakened the movement."),
    # ("What external factors led to the failure of the Khilafat Movement?", "Mustafa Kemal Atatürk abolished the Caliphate in 1924, and British repression stifled momentum."),
    # ("How did the Khilafat and Hijrat Movements shape Muslim political consciousness in British India?", "They fostered political mobilization, highlighted Hindu-Muslim relations, and laid the groundwork for the Pakistan Movement."),
    # ("What socio-economic consequences did the Hijrat Movement have on Indian Muslims?", "Mass migration led to economic hardships, loss of livelihoods, and psychological disillusionment."),
    # ("How did the Khilafat Movement contribute to the foundation of Pakistan?", "It reinforced the distinct political and religious identity of Muslims, influencing the Pakistan Movement."),
]

for question, expected_answer in qa_pairs:
    dense_docs = get_dense_emds_docs(dense_emd_db, emb_model, question, k=5)
    bm25_docs = get_bm25_results_docs(bm25_index, question, texts, metadata, k=5)

    reciprocal_rank_docs = fuse_search_results(dense_emd_db, emb_model, bm25_index, question, texts, metadata, k=5)

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

    #create a new file for questio, answer, and promt
    with open("retrievals.txt", "a") as f:
        f.write(f"Question: {question}\n")
        f.write(f"Expected Answer: {expected_answer}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write("\n")