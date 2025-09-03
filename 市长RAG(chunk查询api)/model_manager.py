# 使用最新的langchain-openai导入
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from cohere import Client as CohereClient
from settings import settings


class ModelManager:
    def __init__(self):
        """初始化所有模型"""
        # 1. 嵌入模型（text-embedding-3-small）
        self.embedding_model = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url
        )

        # 2. 大模型（gpt-4-0613）
        self.llm = ChatOpenAI(
            model_name=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url
        )

        # 3. Cohere重排序模型
        self.cohere_client = CohereClient(api_key=settings.cohere_api_key)

    def get_embedding(self, text: str) -> list[float]:
        """生成文本嵌入向量"""
        return self.embedding_model.embed_query(text)

    def rerank_documents(self, query: str, documents: list, top_k: int = None) -> list:
        """Cohere重排序：优化初步检索结果"""
        if not documents:
            return []
        top_k = top_k or settings.rerank_top_k
        # 提取文档内容（Cohere需要纯文本列表）
        doc_texts = [doc.page_content for doc in documents]
        # 调用重排序API
        response = self.cohere_client.rerank(
            query=query,
            documents=doc_texts,
            top_n=top_k,
            model=settings.reranker_model
        )
        # 按重排序结果返回Document对象
        return [documents[res.index] for res in response.results]