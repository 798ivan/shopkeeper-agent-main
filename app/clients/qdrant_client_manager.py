"""
Qdrant 客户端管理器

统一创建和管理 Qdrant 异步客户端
主要用于保存字段和指标的向量索引，支撑后续问数流程中的语义召回
"""

import asyncio
import random
from typing import Optional

from qdrant_client import AsyncQdrantClient, models

from app.conf.app_config import QdrantConfig, app_config


class QdrantClientManager:
    """管理 Qdrant 客户端的初始化与关闭"""

    def __init__(self, qdrant_config: QdrantConfig):
        # 保存配置对象，后面初始化客户端时要从这里读取 host 和 port
        self.qdrant_config = qdrant_config
        # 先把 client 声明出来，真正初始化放到 init() 中进行
        self.client: Optional[AsyncQdrantClient] = None

    def _get_url(self) -> str:
        """拼接 Qdrant 服务地址"""
        return f"http://{self.qdrant_config.host}:{self.qdrant_config.port}"

    def init(self):
        """
        显式初始化 Qdrant 客户端
        这里不在 __init__ 中直接初始化，是为了和项目的生命周期管理保持一致
        """
        self.client = AsyncQdrantClient(url=self._get_url())

    async def close(self):
        """关闭 Qdrant 客户端连接"""
        await self.client.close()


# 创建一个全局的管理器对象
# 后续项目中的其他模块都通过它来获取同一套 Qdrant 客户端
qdrant_client_manager = QdrantClientManager(app_config.qdrant)

# 判断当前脚本是否作为主程序运行（而不是被其他模块导入）
if __name__ == "__main__":
    # ============ 初始化阶段 ============
    # 先初始化客户端，后面的测试逻辑才能真正访问 Qdrant
    # qdrant_client_manager 是一个自定义的管理器对象，负责管理 Qdrant 客户端的生命周期
    # init() 方法会建立与 Qdrant 服务的连接，设置好连接参数
    qdrant_client_manager.init()


    # ============ 定义测试协程 ============
    async def test():
        """
        执行一次集合创建、写入和查询，验证 Qdrant 接入链路是否正常工作

        这是一个完整的端到端测试流程：
        1. 创建集合（如果不存在）
        2. 批量写入100个随机向量
        3. 用随机向量进行相似度检索
        4. 打印结果
        """

        # 从管理器中获取已初始化的 Qdrant 异步客户端实例
        # 这样就不用每次重新创建客户端连接了
        client = qdrant_client_manager.client

        # ============ 第一步：创建集合 ============
        # 检查集合 "my_collection" 是否已存在
        # collection_exists() 返回布尔值：True表示存在，False表示不存在
        if not await client.collection_exists("my_collection"):
            # 如果集合不存在，则创建一个新的集合
            await client.create_collection(
                collection_name="my_collection",  # 集合名称，类似于MySQL的表名
                vectors_config=models.VectorParams(
                    # 当前集合中的向量维度是 10
                    # 这意味着所有插入这个集合的向量都必须是10维的
                    # 维度大小一旦设定就不能修改，除非删除重建集合
                    size=10,

                    # 使用余弦相似度作为距离计算方式
                    # COSINE: 计算两个向量之间的夹角余弦值
                    # 取值范围：[-1, 1]，值越接近1表示越相似
                    # 其他可选值：EUCLID（欧氏距离）、DOT（点积）等
                    distance=models.Distance.COSINE,
                ),
            )
            # 注意：这里只创建了集合，但没有指定payload schema
            # 这意味着payload可以存储任意JSON数据，非常灵活

        # ============ 第二步：批量写入向量数据 ============
        # 向集合中写入 100 个随机 point
        # 每个 point 都有一个 id 和一个 10 维向量
        # 使用列表推导式批量生成100个随机向量
        await client.upsert(
            collection_name="my_collection",  # 指定要写入的集合
            points=[
                # 为每个i创建一个PointStruct对象
                models.PointStruct(
                    id=i,  # 使用整数作为唯一标识符，从0到99
                    # 生成一个包含10个随机浮点数的列表
                    # random.random() 生成 [0.0, 1.0) 之间的随机数
                    # 每个向量的每个维度都是随机值，模拟真实数据
                    vector=[random.random() for _ in range(10)],
                )
                # 循环生成100个点（i从0到99）
                for i in range(100)
            ],
        )
        # upsert = update + insert
        # 如果id已存在则更新，不存在则插入
        # 这里因为是新集合，所以全部是插入操作

        # ============ 第三步：执行向量相似度检索 ============
        # 用一个随机生成的查询向量做相似度检索
        # 这个查询向量也是10维的，与集合中的向量维度一致
        res = await client.query_points(
            collection_name="my_collection",  # 指定要查询的集合

            # 查询向量：生成一个10维的随机向量
            # type: ignore 是类型检查器的忽略标记，因为random.random()返回float
            # 但类型检查器可能预期更具体的类型
            query=[random.random() for _ in range(10)],  # type: ignore

            limit=10,  # 最多返回10条结果（按相似度从高到低排序）

            # score_threshold=0.8 表示只保留分数不低于0.8的结果
            # 这是一个过滤阈值，可以过滤掉相似度不高的结果
            # 例如：如果所有结果的分数都低于0.8，则返回空列表
            # 如果不设置此参数，则返回所有结果（受limit限制）
            score_threshold=0.8,
        )

        # ============ 第四步：打印查询结果 ============
        # 打印查询结果对象
        # res 是一个 QueryResponse 对象，包含：
        # - points: 匹配的点列表，每个点包含id、score、payload等信息
        # - 如果没有匹配的点，points为空列表
        print(res)


    # ============ 执行测试协程 ============
    # asyncio.run() 是Python异步程序的启动方式
    # 它会：
    # 1. 创建一个新的事件循环（Event Loop）
    # 2. 将test()协程作为任务提交到事件循环
    # 3. 运行事件循环直到test()完成
    # 4. 关闭事件循环并清理资源
    asyncio.run(test())


# if __name__ == "__main__":
#     # 先初始化客户端，后面的测试逻辑才能真正访问 Qdrant
#     qdrant_client_manager.init()
#
#     async def test():
#         """执行一次集合创建、写入和查询，验证 Qdrant 接入链路"""
#         client = qdrant_client_manager.client
#         # 如果集合不存在，就先创建一个集合
#         if not await client.collection_exists("my_collection"):
#             await client.create_collection(
#                 collection_name="my_collection",
#                 vectors_config=models.VectorParams(
#                     # 当前集合中的向量维度是 10
#                     size=10,
#                     # 使用余弦相似度作为距离计算方式
#                     distance=models.Distance.COSINE,
#                 ),
#             )
#
#         # 向集合中写入 100 个随机 point
#         # 每个 point 都有一个 id 和一个 10 维向量
#         await client.upsert(
#             collection_name="my_collection",
#             points=[
#                 models.PointStruct(
#                     id=i,
#                     vector=[random.random() for _ in range(10)],
#                 )
#                 for i in range(100)
#             ],
#         )
#
#         # 用一个随机生成的查询向量做相似度检索
#         # limit=10 表示最多返回 10 条结果
#         # score_threshold=0.8 表示只保留分数不低于 0.8 的结果
#         res = await client.query_points(
#             collection_name="my_collection",
#             query=[random.random() for _ in range(10)],  # type: ignore
#             limit=10,
#             score_threshold=0.8,
#         )
#
#         print(res)
#
#     asyncio.run(test())
