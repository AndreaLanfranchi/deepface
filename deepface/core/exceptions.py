
class MissingOptionalDependencyError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class InsufficentVersionRequirementError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class FaceNotFoundError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)