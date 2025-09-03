from fastapi import FastAPI, HTTPException
from typing import List, Dict
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# 本地模块导入
from data_handlers import MongoDBAtlasHandler, DocumentProcessor
from model_manager import ModelManager
from rag_system import RerankRetriever

# 全局检索器实例
retriever = None


# 数据模型定义
class QueryRequest(BaseModel):
    query: str
    top_k: int = None


class ChunkResponse(BaseModel):
    chunk_idx: int
    source: str
    content: str
    total_chunks: int
    similarity_score: float = None  # 相似度分数


class QueryResponse(BaseModel):
    query: str
    chunk_count: int
    relevant_chunks: List[ChunkResponse]


def initialize_retriever(force_reload: bool = False) -> RerankRetriever:
    """初始化检索器"""
    mongo_handler = MongoDBAtlasHandler()
    model_manager = ModelManager()
    # 知识库
    doc_processor = DocumentProcessor(mongo_handler, model_manager)
    if force_reload or mongo_handler.doc_collection.count_documents({}) == 0:
        doc_processor.init_knowledge_base(force_reprocess=True)
        print("无已有知识库和分片，需新建")
    else:
        print("✅ 检测到已有知识库和分片，使用现有数据")

    return RerankRetriever(mongo_handler, model_manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    if retriever is None:
        print("🔄 正在初始化检索系统...")
        retriever = initialize_retriever(force_reload=False)
        print("✅ 检索系统初始化完成")
    yield  # 应用运行中
    # 关闭时清理连接
    if retriever and hasattr(retriever.mongo_handler, 'close_connection'):
        retriever.mongo_handler.close_connection()
        print("🔌 MongoDB连接已释放")


# 初始化FastAPI应用
app = FastAPI(
    title="RAG查询API",
    description="输入query和top_k返回rerank后的候选片段top k",
    lifespan=lifespan
)


@app.post("/query_chunks", response_model=QueryResponse, summary="查询相关片段")
async def query_chunks(request: QueryRequest):
    """根据输入的查询内容，返回处理后的候选片段top k"""
    if not retriever:
        raise HTTPException(status_code=500, detail="系统未初始化完成")

    try:
        # 直接调用检索器获取片段
        chunks = retriever.get_relevant_chunks(request.query, request.top_k)

        return {
            "query": request.query,
            "chunk_count": len(chunks),
            "relevant_chunks": [
                {
                    "chunk_idx": chunk.metadata.get("chunk_idx"),
                    "source": chunk.metadata.get("source"),
                    "content": chunk.page_content,
                    "total_chunks": chunk.metadata.get("total_chunks"),
                    "similarity_score": chunk.metadata.get("similarity_score")
                } for chunk in chunks
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")


@app.get("/health", summary="健康检查")
async def health_check():
    """检查服务是否正常运行"""
    return {
        "status": "healthy",
        "service": "Chunk Retrieval API",
        "initialized": retriever is not None
    }


if __name__ == "__main__":
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        workers=1
    )