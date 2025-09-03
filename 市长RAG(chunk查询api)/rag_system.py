from langchain.docstore.document import Document
from typing import List, Dict
from settings import settings


class RerankRetriever:
    """向量检索→Cohere重排序"""

    def __init__(self, mongo_handler, model_manager):
        self.mongo_handler = mongo_handler
        self.model_manager = model_manager

    def get_relevant_chunks(self, query: str, top_k: int = None) -> List[Document]:
        """输入query，返回重排序后的top k片段"""
        # 1. 生成查询向量
        query_emb = self.model_manager.get_embedding(query)
        # 2. MongoDB向量检索
        similar_vecs = self.mongo_handler.retrieve_similar_vectors(
            query_embedding=query_emb,
            top_k=top_k or settings.retrieval_top_k
        )
        # 3. 转换为LangChain Document对象
        raw_docs = [
            Document(
                page_content=vec["chunk_text"],
                metadata={** vec["metadata"], "similarity_score": vec.get("similarity_score")}
            ) for vec in similar_vecs
        ]
        # 4. Cohere重排序
        return self.model_manager.rerank_documents(query, raw_docs)