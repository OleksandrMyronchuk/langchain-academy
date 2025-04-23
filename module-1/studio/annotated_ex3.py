from typing import Annotated, get_type_hints
from dataclasses import dataclass, fields, asdict

# 1. Define a marker
class JsonField:
    pass

# 2. Dataclass with annotations
@dataclass
class Person:
    name: Annotated[str, JsonField()]
    password: str
    email: Annotated[str, JsonField()]

# 3. Serializer that respects JsonField
def to_json(obj) -> dict:
    result = {}
    hints = get_type_hints(obj.__class__, include_extras=True)
    for f in fields(obj):
        ann = hints.get(f.name)
        if hasattr(ann, "__metadata__") and any(isinstance(m, JsonField) for m in ann.__metadata__):
            result[f.name] = getattr(obj, f.name)
    return result

# 4. Usage
p = Person(name="Alice", password="secret", email="alice@example.com")
print(to_json(p))
# → {'name': 'Alice', 'email': 'alice@example.com'}
