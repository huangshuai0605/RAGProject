import multiprocessing
import time
import random
from multiprocessing import Queue

#生产者，生产数据
def producer(queue, name):
    """生产者进程：生成数字并放入队列"""
    print(f"[生产者{name}] 开始工作")

    for i in range(5):   #生成5个数据
        time.sleep(random.random())
        data = f"数据{i}"
        queue.put(data)
        print(f"[生产者{name}] 生产了:{data}")
    queue.put(None)
    print(f"[生产者{name}] 工作完成")

# 消费者：处理数据
def consumer(queue, name):
    """消费者进程：从队列取出并处理"""
    print(f"[消费者{name}] 就绪")

    while True:
        item = queue.get()  # 等待数据

        if item is None:  # 收到结束信号
            queue.put(None)  # 传递信号（如果有多个消费者）
            break

        time.sleep(random.random() * 2)  # 模拟处理时间
        print(f"[消费者{name}] 处理了: {item}")

    print(f"[消费者{name}] 完成工作")


if __name__ == '__main__':
    # 创建队列
    q = Queue()

    # 创建进程
    p1 = multiprocessing.Process(target=producer, args=(q, "A"))
    c1 = multiprocessing.Process(target=consumer, args=(q, "X"))

    # 启动进程
    p1.start()
    c1.start()

    # 等待进程结束
    p1.join()
    c1.join()

    print("\n✅ 所有任务完成！")