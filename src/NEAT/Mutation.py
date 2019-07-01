from src.NEAT.Connection import Connection


class NodeMutation:

    def __init__(self, node_id, from_conn, to_conn, old_conn_innov):
        self.node_id = node_id
        self.from_conn: Connection = from_conn
        self.to_conn: Connection = to_conn
        self.old_conn_innov = old_conn_innov

    def get_lookup(self):
        return str(self.old_conn_innov)

    def __eq__(self, other):
        if type(other) != NodeMutation:
            return False
        return other.node_id == self.node_id and other.from_conn == self.from_conn and other.to_conn == self.to_conn

    def __hash__(self):
        s = str(self.node_id) + ':' + \
            str(self.from_conn.from_node.id) + ':' + \
            str(self.to_conn.to_node.id)

        return hash(s)


class ConnectionMutation:

    def __init__(self, conn):
        self.conn: Connection = conn

    def get_lookup(self):
        return str(self.conn.from_node.id) + ':' + str(self.conn.to_node.id)

    def __eq__(self, other):
        if type(other) != ConnectionMutation:
            return False

        return other.conn == self.conn

    def __hash__(self):
        return self.conn.innovation
