import catalogue
import confection


class Registry(confection.registry):
    models = catalogue.create("osc", "llm", "models", entry_points=True)
    chat_templates = catalogue.create("osc", "llm", "chat_templates", entry_points=True)

    @classmethod
    def create(cls, registry_name: str, entry_points: bool = False) -> None:
        """Create a new custom registry."""
        if hasattr(cls, registry_name):
            raise ValueError(f"Registry '{registry_name}' already exists")
        reg = catalogue.create("osc", registry_name, entry_points=entry_points)
        setattr(cls, registry_name, reg)


__all__ = ["Registry"]
