# Mayor_RAG
Preliminary RAG system for Mayor in aivilization.

一.代码结构
1. .env
包括Mongodb连接配置，所有模型的api配置

2. data_handlers.py
  class MongoDBAtlasHandler：包含mongodb的连接、存储、读取，和在本地用余弦相似度进行初步检索（现有的Mongodb Atlas似乎不支持向量检索）
  class DocumentProcessor：包含加载本地文档的功能（现在基本用不到）
  P.S. :最开始把文档分片存入mongodb时有用到llamaparse 

3. model_manager.py
class ModelManager: 用于初始化所有模型

4. rag_system.py
这里只保留了向量检索和reranker部分
class RerankRetriever: 包含向量检索和reranker部分
5. main_api.py
包含api实现与检查服务器是否正常运行
二. 输入输出示例
输入：
  {
    "query": "如何打造晶体管",
    "top_k": 3
  }

输出
<img width="1030" height="799" alt="image" src="https://github.com/user-attachments/assets/1ee914b5-1947-4015-8963-7a838db8a4c2" />
