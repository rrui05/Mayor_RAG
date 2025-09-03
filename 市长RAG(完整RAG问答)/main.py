from settings import settings
from data_handlers import MongoDBAtlasHandler, DocumentProcessor
from model_manager import ModelManager
from rag_system import RerankRetriever, RAGSystem


def initialize_rag_system(force_reload: bool = False) -> RAGSystem:
    """初始化RAG系统（使用MongoDB Atlas）"""
    # 1. 初始化Atlas连接
    mongo_handler = MongoDBAtlasHandler()
    # 2. 初始化模型管理器
    model_manager = ModelManager()
    # 3. 初始化文档处理器
    doc_processor = DocumentProcessor(mongo_handler, model_manager)
    # 4. 初始化知识库（支持强制重新处理）
    if force_reload or mongo_handler.doc_collection.count_documents({}) == 0:
        doc_processor.init_knowledge_base(force_reprocess=True)
    else:
        print("✅ 检测到已有知识库和分片，使用现有数据")
    # 5. 初始化检索器和RAG系统
    retriever = RerankRetriever(mongo_handler, model_manager)
    return RAGSystem(retriever, model_manager)


def main():
    print("🔄 正在初始化基于MongoDB Atlas的RAG系统...")
    # 如需重新处理文档，将force_reload设为True
    rag_system = initialize_rag_system(force_reload=False)
    print("✅ 系统初始化完成\n")

    # 交互查询
    while True:
        query = input("请输入查询内容（输入'quit'退出）：")
        if query.lower() == "quit":
            print("👋 退出系统")
            break

        print("\n🔍 正在检索相关文档...")
        chunks = rag_system.get_chunk_top_k(query)
        # print(f"✅ 找到{len(chunks)}个相关片段：")
        # for i, chunk in enumerate(chunks, 1):
        #     print(f"\n--- 片段{i} ---")
        #     print(f"来源：{chunk.metadata.get('source', '未知')}")
        #     print(f"内容：{chunk.page_content}")

        print("\n🤖 正在生成回答...")
        result = rag_system.get_answer_with_chunks(query)
        print(f"\n--- 回答 ---")
        print(result["answer"])
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()