normal_chunking_param = {
    "chunk_size": 600,
    "chunk_overlap": 50,
}

sementic_chunking_param = {
    "buffer_size": 1,
    "breakpoint_threshold_type": "percentile", #options: "standard_deviation", "percentile", "interquartile", "gradient"
    "breakpoint_threshold_amount": 95,
}

bm25_param = {
    "k1": 1.5,
    "b": 0.75,
    "epsilon": 0.25,
}

dense_embedding_param = {
    "model_name": "BAAI/bge-small-en",
    "distance_strategy": "cosine", # probably dont need to change this
    # options: "euclidean", "max_inner_product", "dot_product", "jaccard", "cosine"
    "index_path": "dense_index" #this is just for path so no need to experiement with this
}

reciprocal_rank_fusion_param = {
    "k": 60
}

