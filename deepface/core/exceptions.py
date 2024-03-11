
class MissingOptionalDependency(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class InsufficentVersionRequirement(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
