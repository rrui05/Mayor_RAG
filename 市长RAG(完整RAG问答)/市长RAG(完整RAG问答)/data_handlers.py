from pymongo import MongoClient
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import datetime
import os
from settings import settings


class MongoDBAtlasHandler:
    def __init__(self, use_chinese_collection: bool = False):
        """初始化连接，支持切换向量集合"""
        try:
            self.client = MongoClient(settings.mongo_uri)
            self.client.admin.command('ping')
            print(f"✅ 成功连接MongoDB Atlas：{settings.mongo_uri.split('@')[-1]}")
        except Exception as e:
            raise ConnectionError(f"❌ MongoDB连接失败：{str(e)}") from e

        self.db = self.client[settings.mongo_db_name]
        self.doc_collection = self.db[settings.mongo_collection_name]
        # 根据标志选择向量集合
        if use_chinese_collection:
            self.vector_collection = self.db[settings.mongo_vector_collection_chinese]
        else:
            self.vector_collection = self.db[settings.mongo_vector_collection]

    def _create_vector_index(self):
        """移除向量索引创建逻辑，在本地进行向量检索"""
        pass

    def store_document(self, document: Document) -> str:
        """存储完整文档到Atlas"""
        doc_data = {
            "page_content": document.page_content,
            "metadata": document.metadata,
            "created_at": datetime.datetime.utcnow()
        }
        insert_result = self.doc_collection.insert_one(doc_data)
        return str(insert_result.inserted_id)

    def store_embedding(self, doc_id: str, embedding: List[float], chunk_text: str, chunk_metadata: Dict):
        """存储文档片段和向量到Atlas"""
        embedding_data = {
            "doc_id": doc_id,
            "embedding": embedding,  # 1536维向量
            "chunk_text": chunk_text,
            "metadata": chunk_metadata,
            "created_at": datetime.datetime.utcnow()
        }
        self.vector_collection.insert_one(embedding_data)

    def retrieve_similar_vectors(self, query_embedding: List[float], top_k: int = None) -> List[Dict]:
        """使用本地余弦相似度进行向量检索"""
        top_k = top_k or settings.retrieval_top_k
        try:
            # 获取所有文档的向量
            all_docs = list(
                self.vector_collection.find({}, {'embedding': 1, 'chunk_text': 1, 'metadata': 1, 'doc_id': 1}))

            if not all_docs:
                return []

            # 构建向量矩阵
            embeddings_matrix = np.array([doc['embedding'] for doc in all_docs])
            query_embedding = np.array(query_embedding).reshape(1, -1)

            # 计算余弦相似度
            similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]

            # 获取top_k个最相似的文档索引
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            # 构建结果
            results = []
            for idx in top_indices:
                doc = all_docs[idx]
                similarity_score = float(similarities[idx])
                results.append({
                    **doc,
                    "similarity_score": similarity_score,  # 确保这个字段存在
                    "chunk_text": doc["chunk_text"],
                    "metadata": {**doc["metadata"], "similarity_score": similarity_score}  # 存入metadata
                })
            return results

        except Exception as e:
            print(f"❌ 向量检索失败：{str(e)}")
            return []

    def get_existing_chunks(self, doc_id: str) -> List[Dict]:
        """从数据库读取指定文档的已有chunks"""
        return list(self.vector_collection.find(
            {"doc_id": doc_id},
            {"chunk_text": 1, "metadata": 1, "embedding": 1, "_id": 0}
        ))

    def close_connection(self):
        """关闭Atlas连接"""
        self.client.close()
        print("🔌 MongoDB Atlas连接已关闭")


class DocumentProcessor:
    def __init__(self, mongo_handler: MongoDBAtlasHandler, model_manager):
        """文档处理器（支持复用已有chunks）"""
        self.mongo_handler = mongo_handler
        self.model_manager = model_manager
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n"]
        )

    def load_md_document(self, file_path: str) -> Document:
        """加载本地Markdown文档"""
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_path))

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ 本地文件不存在：{file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            raise ValueError(f"❌ 文件编码错误，请使用UTF-8：{file_path}")

        return Document(
            page_content=content,
            metadata={
                "source": file_path,
                "format": "markdown",
                "loaded_at": str(datetime.datetime.utcnow())
            }
        )

    def process_and_store(self, document: Document, force_reprocess: bool = False) -> str:
        """处理并存储文档到Atlas，支持强制重新处理"""
        # 检查文档是否已存在
        existing_doc = self.mongo_handler.doc_collection.find_one({
            "metadata.source": document.metadata["source"]
        })

        if existing_doc and not force_reprocess:
            doc_id = str(existing_doc["_id"])
            # 检查是否已有chunks
            existing_chunks = self.mongo_handler.get_existing_chunks(doc_id)
            if existing_chunks:
                print(f"📄 文档及分片已存在于Atlas，ID：{doc_id}，跳过处理")
                return doc_id

        # 新文档或强制重新处理时执行分片
        doc_id = self.mongo_handler.store_document(document)
        print(f"📄 文档已存储到Atlas，ID：{doc_id}")

        chunks = self.text_splitter.split_documents([document])
        print(f"✂️ 分割为 {len(chunks)} 个片段")

        for chunk_idx, chunk in enumerate(chunks):
            chunk_embedding = self.model_manager.get_embedding(chunk.page_content)
            chunk_metadata = {
                **chunk.metadata,
                "chunk_idx": chunk_idx,
                "total_chunks": len(chunks)
            }
            self.mongo_handler.store_embedding(
                doc_id=doc_id,
                embedding=chunk_embedding,
                chunk_text=chunk.page_content,
                chunk_metadata=chunk_metadata
            )

        return doc_id

    def init_knowledge_base(self, force_reprocess: bool = False):
        """初始化知识库，支持强制重新处理"""
        print(f"\n🔧 初始化知识库：{settings.knowledge_base_path}")

        existing_docs = self.mongo_handler.doc_collection.count_documents({
            "metadata.source": settings.knowledge_base_path
        })

        if existing_docs > 0 and not force_reprocess:
            # 检查是否已有chunks
            doc_id = str(self.mongo_handler.doc_collection.find_one(
                {"metadata.source": settings.knowledge_base_path})["_id"])
            existing_chunks = self.mongo_handler.get_existing_chunks(doc_id)
            if existing_chunks:
                print(f"✅ 知识库及分片已存在于Atlas，跳过加载")
                return

        try:
            md_document = self.load_md_document(settings.knowledge_base_path)
            self.process_and_store(md_document, force_reprocess=force_reprocess)
            print("✅ 知识库已上传到Atlas！")
        except Exception as e:
            print(f"❌ 知识库初始化失败：{str(e)}")
            raise

    def init_chinese_knowledge_base(self, force_reprocess: bool = False):
        """初始化中文知识库"""
        print(f"\n🔧 初始化中文知识库：{settings.knowledge_base_path_chinese}")

        existing_docs = self.mongo_handler.doc_collection.count_documents({
            "metadata.source": settings.knowledge_base_path_chinese
        })

        if existing_docs > 0 and not force_reprocess:
            doc_id = str(self.mongo_handler.doc_collection.find_one(
                {"metadata.source": settings.knowledge_base_path_chinese})["_id"])
            existing_chunks = self.mongo_handler.get_existing_chunks(doc_id)
            if existing_chunks:
                print(f"✅ 中文知识库及分片已存在，跳过加载")
                return

        try:
            md_document = self.load_md_document(settings.knowledge_base_path_chinese)
            self.process_and_store(md_document, force_reprocess=force_reprocess)
            print("✅ 中文知识库已上传到Atlas！")
        except Exception as e:
            print(f"❌ 中文知识库初始化失败：{str(e)}")
            raise