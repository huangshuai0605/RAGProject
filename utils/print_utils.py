# def _print_event(event: dict, _printed: set, max_length=1500):
#     """
#     打印事件信息，特别是对话状态和消息内容。如果消息内容过长，会进行截断处理以保证输出的可读性。
#
#     参数:
#         event (dict): 事件字典，包含对话状态和消息。
#         _printed (set): 已打印消息的集合，用于避免重复打印。
#         max_length (int): 消息的最大长度，超过此长度将被截断。默认值为1500。
#     """
#     current_state = event.get("dialog_state")
#     if current_state:
#         print("当前处于: ", current_state[-1])  # 输出当前的对话状态
#     message = event.get("messages")
#     if message:
#         if isinstance(message, list):
#             message = message[-1]  # 如果消息是列表，则取最后一个
#         if message.id not in _printed:
#             msg_repr = message.pretty_repr(html=True)
#             if len(msg_repr) > max_length:
#                 msg_repr = msg_repr[:max_length] + " ... （已截断）"  # 超过最大长度则截断
#             print(msg_repr)  # 输出消息的表示形式
#             _printed.add(message.id)  # 将消息ID添加到已打印集合中
#
# if __name__ == "__main__":
#     # 初始化已打印集合
#     printed_messages = set()
#     # 示例事件1
#     event1 = {
#         "dialog_state": ["问候", "询问需求"],
#         "messages": [Message(id=1, content="你好，有什么可以帮您？")]
#     }
#     _print_event(event1, printed_messages)

# class Message:
#     """简单的消息类，用于测试"""
#
#     def __init__(self, id: int, content: str):
#         self.id = id
#         self.content = content
#
#     def pretty_repr(self, html: bool = False) -> str:
#         """返回消息的格式化表示"""
#         if html:
#             return f"<div>{self.content}</div>"
#         return self.content


def _print_event(event: dict, _printed: set, max_length=1500):
    """
    打印事件信息，特别是对话状态和消息内容。如果消息内容过长，会进行截断处理以保证输出的可读性。

    参数:
        event (dict): 事件字典，包含对话状态和消息。
        _printed (set): 已打印消息的集合，用于避免重复打印。
        max_length (int): 消息的最大长度，超过此长度将被截断。默认值为1500。
    """
    current_state = event.get("dialog_state")
    if current_state:
        print("当前处于: ", current_state[-1])  # 输出当前的对话状态
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]  # 如果消息是列表，则取最后一个
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... （已截断）"  # 超过最大长度则截断
            print(msg_repr)  # 输出消息的表示形式
            _printed.add(message.id)  # 将消息ID添加到已打印集合中


if __name__ == "__main__":
    # 初始化已打印集合
    printed_messages = set()

    # 示例事件1
    event1 = {
        "dialog_state": ["问候", "询问需求"],
        "messages": [Message(id=1, content="你好，有什么可以帮您？")]
    }
    _print_event(event1, printed_messages)

    # 示例事件2 - 测试长消息截断
    long_content = "这是一条非常长的消息，" * 100
    event2 = {
        "dialog_state": ["询问需求", "提供解决方案"],
        "messages": [Message(id=2, content=long_content)]
    }
    _print_event(event2, printed_messages)

    # 示例事件3 - 测试重复消息不打印
    event3 = {
        "dialog_state": ["结束对话"],
        "messages": [Message(id=1, content="你好，有什么可以帮您？")]
    }
    _print_event(event3, printed_messages)
    print("注意：ID=1的消息不会重复打印")

