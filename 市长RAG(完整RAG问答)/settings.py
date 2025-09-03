import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MongoDB Atlas配置（从.env获取）
    mongo_uri: str = os.getenv("MONGO_URI")
    mongo_db_name: str = os.getenv("MONGO_DB_NAME")
    mongo_collection_name: str = os.getenv("MONGO_COLLECTION_NAME")
    mongo_vector_collection: str = os.getenv("MONGO_VECTOR_COLLECTION")

    # 嵌入模型配置（从.env获取）
    embedding_model: str = os.getenv("EMBEDDING_MODEL")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY")
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1536"))  # 保留默认值作为 fallback

    # LLM 模型配置（从.env获取）
    llm_model: str = os.getenv("LLM_MODEL")
    llm_api_key: str = os.getenv("LLM_API_KEY")
    llm_base_url: str = os.getenv("LLM_BASE_URL")

    # Cohere 重排序模型配置（从.env获取）
    cohere_api_key: str = os.getenv("COHERE_API_KEY")
    reranker_model: str = os.getenv("RERANKER_MODEL")

    # 文档分割配置（从.env获取）
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "20"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))

    # LlamaParse 配置（从.env获取）
    llama_parse_api_key: str = os.getenv("LLAMA_PARSE_API_KEY")

    # 知识库路径（从.env获取）
    knowledge_base_path: str = os.getenv("KNOWLEDGE_BASE_PATH")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()