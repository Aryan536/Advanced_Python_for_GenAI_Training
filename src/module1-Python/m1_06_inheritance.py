# Real-world example: the "common interface, many models" pattern used by
# LangChain / LlamaIndex. Every model class exposes the same generate() method.
class BaseLLM:
    def __init__(self, name, temperature=0.0):
        self.name = name
        self.temperature = temperature

    def generate(self, prompt):
        raise NotImplementedError("Child classes must implement generate()")

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', temperature={self.temperature})"

class EchoLLM(BaseLLM):
    """A fake model for testing pipelines without paying for API calls."""
    def generate(self, prompt):
        return f"[{self.name}] You said: {prompt}"

class FAQBot(BaseLLM):
    """A rule-based 'model' that answers from a small knowledge base."""
    def __init__(self, name, faq):
        super().__init__(name)
        self.faq = faq

    def generate(self, prompt):
        for keyword, answer in self.faq.items():
            if keyword in prompt.lower():
                return f"[{self.name}] {answer}"
        return f"[{self.name}] Sorry, I will connect you to a human agent."

faq = {"refund": "Refunds reach your account in 5 business days.",
       "cancel": "You can cancel before the order is shipped."}

models = [EchoLLM("echo-v1"), FAQBot("faq-bot", faq)]
for model in models:                       # polymorphism: same call, different behaviour
    print(model)
    print("  ->", model.generate("How long does a refund take?"))
