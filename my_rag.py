import os

from embeddings_lookup import fuse_search_results, get_bm25_results_docs, get_dense_emds_docs, load_bm25_index, load_dense_embeddings
from my_chunking_sys import get_rec_split_chunks, get_sementic_embedding_chunks
from my_data_loader import load_markdowns, load_texts
from hyper_parameters import normal_chunking_param, sementic_chunking_param, bm25_param, dense_embedding_param, reciprocal_rank_fusion_param

file_paths = [f for f in os.listdir("doc2_text") if f.endswith(".txt")]
file_paths = [os.path.join("doc2_text", f) for f in file_paths]
docs = load_texts(file_paths)

#chunks = get_sementic_embedding_chunks(docs, **sementic_chunking_param)
chunks = get_rec_split_chunks(docs, **normal_chunking_param)

chunks = chunks

texts = [chunk.page_content for chunk in chunks]
metadata = [chunk.metadata for chunk in chunks]

dense_emd_db, emb_model = load_dense_embeddings(chunks, **dense_embedding_param)
bm25_index = load_bm25_index(chunks)

#format retrivals and use it in prompt
def format_response(doc):
    return f"Page {doc.metadata.get('page', 'Unknown')}: {doc.page_content.strip()}"

q_and_a_tuples = [
    ("How important was Jinnah's 14 points to the creation of Pakistan ?",
     "Jinnah's 14 Points were extremely important in the lead-up to the creation of Pakistan. They served as a concrete articulation of Muslim political demands at a crucial juncture when the Nehru Report had failed to address these concerns. The 14 Points did the following:\n\n* **Defined Muslim demands:** They clearly outlined key issues such as separate electorates and adequate representation, providing a unified platform for the Muslim League.\n* **Basis for negotiations:** These points became the foundation for all subsequent discussions and negotiations with both the Indian National Congress and the British government.\n* **Foundation for Pakistan's demand:** They laid the ideological groundwork for the Lahore Resolution of 1940, which explicitly demanded a separate Muslim state.\n* **Unified Muslims:** The 14 Points played a significant role in uniting the diverse Muslim population of British India under the banner of the Muslim League.\n* **Strategic tool:** They were instrumental in Jinnah's political strategy during talks with Gandhi and in the rejection of the Cripps Mission when Muslim demands were not adequately met.\n* **Electoral success:** The groundwork laid by the 14 Points contributed to the Muslim League's success in the crucial 1945-46 elections, demonstrating widespread Muslim support for their political agenda.\n* **Historical link to Pakistan:** The eventual creation of Pakistan in 1947 can be directly traced back to the political stance and demands established in these 14 Points."),
    ("What were some of the difficulties faced by Pakistan during its creation phase ?",
     "Pakistan faced numerous significant difficulties during its creation phase, including:\n\n* **Geographical Problems:** The unique geographical division of Pakistan into two wings (East and West Pakistan) separated by 1600 km of Indian territory posed immense challenges for communication, travel, and maintaining national cohesion. The cultural, political, and historical differences between the Bengali-majority East and the ethnically diverse West further exacerbated these issues.\n\n* **Political Problems:** Unlike India, which inherited an established governmental structure, Pakistan had to build its entire administrative framework from scratch. The initial government lacked extensive experience, being largely composed of landowners and civil servants. Furthermore, the dominance of West Pakistan in civil and military roles despite East Pakistan having a larger population created early political tensions. The immense workload and failing health of Jinnah, who took on responsibilities beyond the Governor-General's role, also contributed to the political challenges. Sadly, the early misuse of executive power set a problematic precedent for the nation's political development.\n\n* **Economic Problems:** Pakistan inherited predominantly underdeveloped regions with a largely rural population and limited industrial infrastructure. Key resources like jute production were concentrated in East Pakistan, while processing mills were located elsewhere. The nascent nation faced a severe lack of industry and the economic capacity for growth."),
    ("What Pakistand did in return after china supported pakistan on the Kashmir issue ?",
     "In 1964, after China declared its support for Pakistan on the Kashmir issue, Pakistan supported China’s entry to the UN Security Council."),
    ("What were the notable events in Pakistan and China relations between 1972 to 1990?",
     "Notable events in Pakistan and China relations between 1972 and 1990 included:\n\n* **1972:** Z.A. Bhutto's visit to China where he sought and received commitments for heavy military and economic aid. Chinese assistance was crucial in building the Heavy Mechanical Complex (HMC) in Taxila.\n* **1978:** During Zia-ul-Haq's regime, the historic Silk Route was significantly modernized and reopened as the Karakoram Highway (KKH), a testament to their growing connectivity.\n* **1986:** President Zia visited China and signed a significant nuclear cooperation treaty, marking deeper strategic alignment.\n* **1989:** Pakistan offered moral support to China during the Tiananmen Square protests. Later that year, Chinese Premier Peng's visit to Pakistan further solidified bilateral ties.\n* **Early 1990s:** During Nawaz Sharif's first term as Prime Minister, he visited China in February 1991, followed by a reciprocal visit from Chinese President Mr. Shangkun to Pakistan in October 1991."),
    ("What were the reasons of Pakistan joining UNO in 1947 ?",
     "Pakistan joined the United Nations Organization (UNO) in 1947 for several key reasons:\n\n* **International Recognition:** To gain immediate international legitimacy and recognition as a newly independent sovereign state.\n* **Commitment to Peace:** To demonstrate its commitment to the principles of global peace and cooperation.\n* **Support for Freedom Movements:** To align itself with and support other nations undergoing decolonization and freedom movements, such as those in Indonesia, Morocco, Algeria, and Palestine.\n* **Kashmir Issue:** To have an international platform to raise the Kashmir dispute and seek a resolution.\n* **Economic Aid:** To access international economic assistance and the resources of financial institutions like the World Bank.\n* **Regional Dispute Resolution:** To utilize the UN framework for resolving regional conflicts, such as the Canal Water Dispute with India."),
]

def get_retrievals_file(qa_pairs, dense_docs, bm25_docs, file_name="retrievals.txt", which_emd=None):

    for question, expected_answer in qa_pairs:
        dense_docs = get_dense_emds_docs(dense_emd_db, emb_model, question, k=5)
        bm25_docs = get_bm25_results_docs(bm25_index, question, texts, metadata, k=5)

        reciprocal_rank_docs = fuse_search_results(dense_emd_db, emb_model, bm25_index, question, texts, metadata, k=5)

        retrievals = []

        choice = reciprocal_rank_docs
        if (which_emd != None):
            choice = bm25_docs if which_emd == "bm25" else dense_docs

        for doc, _ in choice:
            retrievals.append(format_response(doc))

        #make a prompt for llm using retrievals and question
        retrieval = f"""
            ### **Retrieved Information**:
            1. {retrievals[0]}

            2. {retrievals[1]}

            3. {retrievals[2]}

            4. {retrievals[3]}

            5. {retrievals[4]}
        """

        prompt = f"""
        You are an AI assistant tasked with answering questions based on retrieved knowledge.

        ### **Retrieved Information**:
            1. {retrievals[0]}

            2. {retrievals[1]}

            3. {retrievals[2]}

            4. {retrievals[3]}

            5. {retrievals[4]}

        ### **Question**:
            {question}

        ### **Instructions**:
        - Integrate the key points from all retrieved responses into a **cohesive, well-structured answer**.
        - If the responses are **contradictory**, mention the different perspectives.
        - If none of the retrieved responses contain relevant information, reply:
          **"I couldn't find a good response to your query in the database."**
        """

        #create a new file for questio, answer, and promt
        with open(file_name, "a") as f:
            f.write(f"Question: {question}\n")
            f.write(f"Expected Answer: {expected_answer}\n")
            f.write(f"Retrievals: {retrieval}\n")
            f.write(f"prompt: {prompt}\n")
            f.write("\n")
#rec_chunking --> with reciprocal rank fusion (k=60)
get_retrievals_file(q_and_a_tuples, dense_emd_db, bm25_index, file_name="retrievals_rec_chunk_reciprocal_rank_fusion_k=5")
