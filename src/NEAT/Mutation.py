from src.NEAT.Connection import Connection


class NodeMutation:
    def __init__(self, node_id, from_conn, to_conn):
        self.node_id = node_id
        self.from_conn = from_conn
        self.to_conn = to_conn

    def __eq__(self, other):
        return other.node_id == self.node_id and other.from_conn == self.from_conn and other.to_conn == self.to_conn

    def __hash__(self):
        s = str(self.node_id) + ':' + \
            str(self.from_conn.from_node.id) + ':' + \
            str(self.to_conn.to_node.id)

        return hash(s)


class ConnectionMutation:
    def __init__(self, conn):
        self.conn: Connection = conn

    def __eq__(self, other):
        return other.conn == self.conn

    def __hash__(self):
        s = str(self.conn.from_node) + ':' + str(self.conn.to_node)
        return hash(s)
