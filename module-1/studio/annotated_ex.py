from typing import Annotated, get_type_hints
import inspect

# 1. Define metadata markers
class MinLength:
    def __init__(self, length: int):
        self.length = length

class MaxValue:
    def __init__(self, value: int):
        self.value = value

# 2. Decorator to enforce Annotated constraints
def enforce_annotations(func):
    sig = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)

    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        for name, value in bound.arguments.items():
            annotation = hints.get(name)
            if hasattr(annotation, "__metadata__"):
                for meta in annotation.__metadata__:
                    if isinstance(meta, MinLength) and len(value) < meta.length:
                        raise ValueError(f"'{name}' must have at least {meta.length} characters")
                    if isinstance(meta, MaxValue) and value > meta.value:
                        raise ValueError(f"'{name}' must be ≤ {meta.value}")
        return func(*args, **kwargs)
    return wrapper

# 3. Apply to a function
@enforce_annotations
def register_user(
    username: Annotated[str, MinLength(3)],
    age: Annotated[int, MaxValue(120)]
):
    print(f"Registered {username} age {age}")

# 4. Usage
register_user("abby", 30)   # OK
register_user("Al", 30)   # ValueError: 'username' must have at least 3 characters
#register_user("Charlie", 130)  # ValueError: 'age' must be ≤ 120
