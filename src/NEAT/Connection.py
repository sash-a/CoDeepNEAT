class Connection:
    def __init__(self, from_node, to_node, connection_type=None, enabled=True, innovation=0):
        self.from_node = from_node
        self.to_node = to_node
        self.connection_type = connection_type
        self.enabled = enabled
        self.innovation = innovation

    def __repr__(self):
        return 'from: ' + str(self.from_node) + ' to: ' + str(self.to_node)

    def __eq__(self, other):
        return self.from_node == other.from_node and self.to_node == other.to_node

    def __hash__(self):
        return self.innovation
