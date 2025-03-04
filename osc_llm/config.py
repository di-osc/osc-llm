import catalogue
import confection
from confection import Config


class registry(confection.registry):
    layers = catalogue.create("osc", "layers", entry_points=True)

    architectures = catalogue.create("osc", "architectures", entry_points=True)

    model_helpers = catalogue.create("osc", "model_helpers", entry_points=True)

    quantizers = catalogue.create("osc", "quantizers", entry_points=True)

    chat_templates = catalogue.create("osc", "chat_templates", entry_points=True)

    samplers = catalogue.create("osc", "samplers", entry_points=True)

    engines = catalogue.create("osc", "engines", entry_points=True)

    @classmethod
    def create(cls, registry_name: str, entry_points: bool = False) -> None:
        """Create a new custom registry."""
        if hasattr(cls, registry_name):
            raise ValueError(f"Registry '{registry_name}' already exists")
        reg = catalogue.create("osc", registry_name, entry_points=entry_points)
        setattr(cls, registry_name, reg)


__all__ = ["Config", "registry"]
