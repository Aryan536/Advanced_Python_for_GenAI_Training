from dataclasses import dataclass, field
from typing import List

# Real-world example: managing chat history for an LLM chatbot with a word budget
@dataclass
class Message:
    role: str        # "system", "user" or "assistant"
    content: str

    def word_count(self):
        return len(self.content.split())

@dataclass
class Conversation:
    max_words: int = 30
    messages: List[Message] = field(default_factory=list)

    def add(self, role, content):
        self.messages.append(Message(role, content))
        self._trim()

    def total_words(self):
        return sum(m.word_count() for m in self.messages)

    def _trim(self):
        # Keep the system message (index 0); drop the oldest turns when over budget
        while self.total_words() > self.max_words and len(self.messages) > 2:
            removed = self.messages.pop(1)
            print(f"  (trimmed old message: '{removed.content[:30]}...')")

    def to_api_format(self):
        return [{"role": m.role, "content": m.content} for m in self.messages]

chat = Conversation(max_words=30)
chat.add("system", "You are a helpful e-commerce support assistant.")
chat.add("user", "Where is my order ORD200145? It was due yesterday.")
chat.add("assistant", "Your order is out for delivery and will arrive today by 7 PM.")
chat.add("user", "Can I change the delivery address to my office in Koramangala?")

print("Words in history:", chat.total_words())
for msg in chat.to_api_format():
    print(msg)
