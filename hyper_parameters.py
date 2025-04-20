normal_chunking_param = {
    "chunk_size": 1200, # 600, 800, 1200
    "chunk_overlap": 150, # 30, 75, 150
}

sementic_chunking_param = {
    "buffer_size": 2, #1, 5, 10
    "breakpoint_threshold_type": "gradient", #options: "standard_deviation", "percentile", "interquartile", "gradient"
    "breakpoint_threshold_amount": 99, # 80, 90, 95
}

bm25_param = {
    "k1": 1.5,
    "b": 0.75,
    "epsilon": 0.25,
}

dense_embedding_param = {
    "model_name": "BAAI/bge-small-en", ##BAAI/bge-small-en, #thenlper/gte-small
    "distance_strategy": "cosine", # probably dont need to change this
    # options: "euclidean", "max_inner_product", "dot_product", "jaccard", "cosine"
    "index_path": "dense_index" #this is just for path so no need to experiement with this
}

reciprocal_rank_fusion_param = {
    "k": 60 # 20, 40, 60
}

