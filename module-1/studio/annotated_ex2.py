from typing import Annotated, get_type_hints

# 1. Define an injection marker
class Inject:
    def __init__(self, cls):
        self.cls = cls

# 2. A basic injector/factory
class Injector:
    def __init__(self):
        self._cache = {}

    def create(self, cls):
        # Include extras so Annotated metadata shows up
        hints = get_type_hints(cls.__init__, include_extras=True)

        kwargs = {}
        for param_name, annotation in hints.items():
            # skip 'self'
            if param_name == 'self':
                continue

            metadata = getattr(annotation, "__metadata__", ())
            for meta in metadata:
                if isinstance(meta, Inject):
                    # create or reuse instance of the requested class
                    if meta.cls not in self._cache:
                        self._cache[meta.cls] = meta.cls()
                    kwargs[param_name] = self._cache[meta.cls]

        # instantiate the target, injecting any found dependencies
        return cls(**kwargs)

# 3. Example services and clients
class Database:
    def connect(self):
        print("DB connected")

class Service:
    # annotate the 'db' parameter so Injector knows what to inject
    def __init__(self, db: Annotated[Database, Inject(Database)]):
        self.db = db

    def run(self):
        self.db.connect()

# 4. Usage
if __name__ == "__main__":
    injector = Injector()
    svc = injector.create(Service)
    svc.run()   # prints "DB connected"
