from langchain.docstore.document import Document
from typing import List, Dict
from settings import settings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class RerankRetriever:
    """检索器：向量检索→Cohere重排序"""

    def __init__(self, mongo_handler, model_manager):
        self.mongo_handler = mongo_handler
        self.model_manager = model_manager

    def get_relevant_chunks(self, query: str, top_k: int = None) -> List[Document]:
        """输入query，返回重排序后的top k片段"""
        # 1. 生成查询向量
        query_emb = self.model_manager.get_embedding(query)
        # 2. 原生MongoDB向量检索（使用修复后的方法）
        similar_vecs = self.mongo_handler.retrieve_similar_vectors(
            query_embedding=query_emb,
            top_k=top_k or settings.retrieval_top_k
        )
        # 3. 转换为LangChain Document对象
        raw_docs = [
            Document(
                page_content=vec["chunk_text"],
                metadata=vec["metadata"]
            ) for vec in similar_vecs
        ]
        # 4. Cohere重排序（优化结果）
        return self.model_manager.rerank_documents(query, raw_docs)


class RAGSystem:
    """完整RAG系统：检索→生成回答"""

    def __init__(self, retriever: RerankRetriever, model_manager):
        self.retriever = retriever
        self.model_manager = model_manager
        self.qa_chain = self._build_qa_chain()

    def _build_qa_chain(self) -> LLMChain:
        """构建问答链（基于大模型和提示词）"""
        qa_template = """
你是专业问答助手，仅基于【上下文】回答【问题】，不编造信息。
若上下文不足以回答，直接说“根据现有知识库，无法回答该问题”。

【上下文】：
{context}

【问题】：
{question}

【回答】：
        """
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=qa_template.strip()
        )
        return LLMChain(llm=self.model_manager.llm, prompt=prompt)

    def get_chunk_top_k(self, query: str, top_k: int = None) -> List[Document]:
        """API核心：输入query，返回pipeline后的top k候选片段"""
        return self.retriever.get_relevant_chunks(query, top_k)

    def get_answer_with_chunks(self, query: str) -> Dict:
        """输入query，返回“回答+使用的片段”"""
        # 1. 获取相关片段
        relevant_chunks = self.get_chunk_top_k(query)
        # 2. 构建上下文（拼接片段）
        context = "\n\n---\n\n".join([chunk.page_content for chunk in relevant_chunks])
        # 3. 生成回答
        answer = self.qa_chain.run(context=context, question=query)
        # 4. 返回结构化结果
        return {
            "query": query,
            "answer": answer.strip(),
            "relevant_chunks": [
                {
                    "chunk_idx": chunk.metadata.get("chunk_idx", "未知"),
                    "source": chunk.metadata.get("source", "未知"),
                    "content": chunk.page_content,
                    "total_chunks": chunk.metadata.get("total_chunks", "未知")
                } for chunk in relevant_chunks
            ],
            "chunk_count": len(relevant_chunks)
        }