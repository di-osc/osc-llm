import catalogue
import confection


class Registry(confection.registry):
    models = catalogue.create("osc", "llm", "models", entry_points=True)
    chat_templates = catalogue.create("osc", "llm", "chat_templates", entry_points=True)


__all__ = ["Registry"]
