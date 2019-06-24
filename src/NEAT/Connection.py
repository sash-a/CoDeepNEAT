class Connection:
    def __init__(self, in_node, out_node, connection_type=None, enabled=True, innovation=0):
        self.in_node = in_node
        self.out_node = out_node
        self.connection_type = connection_type
        self.enabled = enabled
        self.innovation = innovation

    def __repr__(self):
        return 'innovation: ' + str(self.innovation)

    def __eq__(self, other):
        return self.in_node == other.in_node and self.out_node == other.out_node

    def __hash__(self):
        return self.innovation
