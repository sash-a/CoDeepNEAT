class NEATNode:
    def __init__(self, id, x):
        self.id = id
        self.x = x

    def midpoint(self, other):
        mn = min(self.x, other.x)
        return mn + abs(self.x - other.x) / 2

    def __eq__(self, other):
        return other.id == self.id and self.x == other.x

    def __hash__(self):
        return self.id

    def __repr__(self):
        return str(self.id)
