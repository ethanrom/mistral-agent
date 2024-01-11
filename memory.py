from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

memory_storage = StreamlitChatMessageHistory(key="chat_messages")
memory = ConversationBufferWindowMemory(memory_key="chat_history", human_prefix="User", chat_memory=memory_storage, k=3)