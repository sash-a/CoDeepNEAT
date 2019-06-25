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


if __name__ == '__main__':
    from src.NEAT.Genotype import Node, Connection

    n1 = Node(1, 0)
    n2 = Node(2, 0)
    n3 = Node(3, 1)

    c1 = Connection(n1, n2, innovation=1)
    c2 = Connection(n1, n3, innovation=2)

    c1_fake = Connection(n1, n2, innovation=1231)
    c2_fake = Connection(n1, n3)

    m = ConnectionMutation(c1)
    s = set()
    s.add(m)

    nm = ConnectionMutation(c1_fake)

    if nm in s:
        print('hash working')

    pr = None
    for m in s:
        if m == nm:
            pr = m
    print('done')
