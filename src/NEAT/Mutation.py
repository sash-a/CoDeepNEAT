class NodeMutation:
    def __init__(self, node_id, from_innov, to_innov):
        self.node_id = node_id
        self.from_innov = from_innov
        self.to_innov = to_innov

    def __eq__(self, other):
        return other.node_id == self.node_id and other.from_innov == self.from_innov and other.to_innov == self.to_innov

    def __hash__(self):
        s = str(self.node_id) + ':' + str(self.from_innov) + ':' + str(self.to_innov)
        return hash(s)


class ConnectionMutation:
    def __init__(self, node1, node2, conn_innov):
        self.node1 = node1
        self.node2 = node2
        self.conn_innov = conn_innov

    def __eq__(self, other):
        return other.node_id == self.node1 and other.from_innov == self.node2 and other.to_innov == self.conn_innov

    def __hash__(self):
        s = str(self.node1) + ':' + str(self.node2) + ':' + str(self.conn_innov)
        return hash(s)
