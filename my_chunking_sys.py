from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_experimental.text_splitter import SemanticChunker
import matplotlib.pyplot as plt
import numpy as np
import re

def get_rec_split_chunks(docs ,chunk_size = 65, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strip_whitespace=True
    )
    return text_splitter.split_documents(docs)


#uses # headers, so not useful if headeres are with **, try making custom one later
def get_md_split_chunks(docs, chunk_size = 65, chunk_overlap=0):
    text_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)


def get_sementic_embedding_chunks(docs, breakpoint_threshold_type='percentile', buffer_size=1, breakpoint_threshold_amount=95):
    # Initialize the HuggingFaceEmbeddings
    embeds = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    # Initialize the SemanticChunker
    chunker = SemanticChunker(embeddings=embeds,
                              buffer_size=buffer_size,
                              breakpoint_threshold_type=breakpoint_threshold_type,
                              breakpoint_threshold_amount=breakpoint_threshold_amount)

    # Create chunks
    chunks = chunker.split_documents(docs)
    
    return chunks

"""following function is taken from https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb"""
def get_sementic_embedding_chunks_custom(text, buffer_size=1, breakpoint_percentile_threshold=95):
    single_sentences_list = re.split(r'(?<=[.?!])\s+', text)
    sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(single_sentences_list)]

    # Check if the sentences list is empty
    # if not sentences:
    #     print("No sentences found in the text.")
    #     return []
    # for i, sentence in enumerate(sentences):
    #     print(f"Index {i}: {sentence}")  # Check if 'combined_sentence' exists for all

    for i in range(len(sentences)):

        # Create a string that will hold the sentences which are joined
        combined_sentence = ''

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]['sentence'] + ' '

        # Add the current sentence
        combined_sentence += sentences[i]['sentence']

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += ' ' + sentences[j]['sentence']

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        # print(combined_sentence)
        sentences[i]['combined_sentence'] = combined_sentence

    embeds = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    embeddings = embeds.embed_documents([x['combined_sentence'] for x in sentences])

    for i, sentence in enumerate(sentences):
        sentence['combined_sentence_embedding'] = embeddings[i]

    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
            
        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            
        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]['distance_to_next'] = distance
        
    plt.plot(distances)

    y_upper_bound = max(distances) + 0.1 * max(distances)
    plt.ylim(0, y_upper_bound)
    plt.xlim(0, len(distances))

    # We need to get the distance threshold that we'll consider an outlier
    # We'll use numpy .percentile() for this
    breakpoint_percentile_threshold = 95
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold) # If you want more chunks, lower the percentile cutoff
    plt.axhline(y=breakpoint_distance_threshold, color='r', linestyle='-')

    # Then we'll see how many distances are actually above this one
    num_distances_above_theshold = len([x for x in distances if x > breakpoint_distance_threshold]) # The amount of distances above your threshold
    plt.text(x=(len(distances)*.01), y=y_upper_bound/50, s=f"{num_distances_above_theshold + 1} Chunks")

    # Then we'll get the index of the distances that are above the threshold. This will tell us where we should split our text
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold] # The indices of those breakpoints on your list

    # Start of the shading and text
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, breakpoint_index in enumerate(indices_above_thresh):
        start_index = 0 if i == 0 else indices_above_thresh[i - 1]
        end_index = breakpoint_index if i < len(indices_above_thresh) - 1 else len(distances)

        plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
        plt.text(x=np.average([start_index, end_index]),
                y=breakpoint_distance_threshold + (y_upper_bound)/ 20,
                s=f"Chunk #{i}", horizontalalignment='center',
                rotation='vertical')

    # # Additional step to shade from the last breakpoint to the end of the dataset
    if indices_above_thresh:
        last_breakpoint = indices_above_thresh[-1]
        if last_breakpoint < len(distances):
            plt.axvspan(last_breakpoint, len(distances), facecolor=colors[len(indices_above_thresh) % len(colors)], alpha=0.25)
            plt.text(x=np.average([last_breakpoint, len(distances)]),
                    y=breakpoint_distance_threshold + (y_upper_bound)/ 20,
                    s=f"Chunk #{i+1}",
                    rotation='vertical')

    plt.title("PG Essay Chunks Based On Embedding Breakpoints")
    plt.xlabel("Index of sentences in essay (Sentence Position)")
    plt.ylabel("Cosine distance between sequential sentences")
    plt.show()
        

    # Initialize the start index
    start_index = 0

    # Create a list to hold the grouped sentences
    chunks = []

    # Iterate through the breakpoints to slice the sentences
    for index in indices_above_thresh:
        # The end index is the current breakpoint
        end_index = index

        # Slice the sentence_dicts from the current start index to the end index
        group = sentences[start_index:end_index + 1]
        combined_text = ' '.join([d['sentence'] for d in group])
        chunks.append(combined_text)
            
        # Update the start index for the next group
        start_index = index + 1

    # The last group, if any sentences remain
    if start_index < len(sentences):
        combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
        chunks.append(combined_text)
        
    # Return the chunks
    return chunks