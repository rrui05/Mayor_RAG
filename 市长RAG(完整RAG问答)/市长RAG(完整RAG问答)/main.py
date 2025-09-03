from settings import settings
from data_handlers import MongoDBAtlasHandler, DocumentProcessor
from model_manager import ModelManager
from rag_system import RerankRetriever, RAGSystem, MultiRerankRetriever


# main.py 修改 initialize_rag_system 函数
def initialize_rag_system(force_reload: bool = False) -> RAGSystem:
    # 模型管理器共享
    model_manager = ModelManager()

    # 1. 英文知识库组件
    english_mongo = MongoDBAtlasHandler(use_chinese_collection=False)
    english_processor = DocumentProcessor(english_mongo, model_manager)
    if force_reload or english_mongo.doc_collection.count_documents({}) == 0:
        english_processor.init_knowledge_base(force_reprocess=True)
    else:
        print("✅ 检测到已有英文知识库，使用现有数据")
    english_retriever = RerankRetriever(english_mongo, model_manager)

    # 2. 中文知识库组件
    chinese_mongo = MongoDBAtlasHandler(use_chinese_collection=True)
    chinese_processor = DocumentProcessor(chinese_mongo, model_manager)
    if force_reload or chinese_mongo.doc_collection.count_documents({
        "metadata.source": settings.knowledge_base_path_chinese
    }) == 0:
        chinese_processor.init_chinese_knowledge_base(force_reprocess=True)
    else:
        print("✅ 检测到已有中文知识库，使用现有数据")
    chinese_retriever = RerankRetriever(chinese_mongo, model_manager)

    # 3. 使用多知识库检索器
    multi_retriever = MultiRerankRetriever(chinese_retriever, english_retriever)
    return RAGSystem(multi_retriever, model_manager)

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
        print(f"✅ 找到{len(chunks)}个相关片段：")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n--- 片段{i} ---")
            print(f"来源：{chunk.metadata.get('source', '未知')}")
            print(f"内容：{chunk.page_content}")

        print("\n🤖 正在生成回答...")
        result = rag_system.get_answer_with_chunks(query)
        print(f"\n--- 回答 ---")
        print(result["answer"])
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()