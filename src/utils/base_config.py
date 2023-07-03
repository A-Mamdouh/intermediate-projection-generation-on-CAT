from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict


@dataclass
class BaseConfig(ABC):
    def to_dict(self) -> Dict[str, Any]:
        # Shamelessly copied from: https://stackoverflow.com/a/64693838
        def custom_asdict_factory(data):
            def convert_value(obj):
                if isinstance(obj, Enum):
                    return obj.value
                if isinstance(obj, Tuple):
                    return ", ".join(str(x) for x in obj)
                return obj
            return {key: convert_value(value) for key, value in data}
        return asdict(self, dict_factory=custom_asdict_factory)

    @abstractmethod
    def from_dict(raw_dict: Dict[str, Any]) -> "BaseConfig":
        pass
