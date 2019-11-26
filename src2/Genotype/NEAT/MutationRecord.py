from typing import Dict, Tuple


class MutationRecords:
    def __init__(self, initial_connection_mutations: Dict[tuple, int], initial_node_mutations: Dict[tuple, int],
                 current_max_node_id, current_max_conn_id):
        """
            Records all mutation in a single run so that no innovation is misused
            simultaneously stores node and connection mutation information
        """

        # maps mutation details to the mutation id
        # for connection the mapping is from (from_node_id, to_node_id) to -> connection_mutation_id
        self.connection_mutations = initial_connection_mutations
        # mapping from (connection id, count) -> node id
        # where count is the number of times a mutation has occurred on this connection
        self.node_mutations = initial_node_mutations

        self._next_node_id = current_max_node_id
        self._next_conn_id = current_max_conn_id

    def __repr__(self):
        return 'Connections: ' + repr(self.connection_mutations) + '\nNodes: ' + repr(self.node_mutations)

    def exists(self, mutation: Tuple[int, int], conn: bool) -> bool:
        """
        Checks if a mutation exists, must specify if it is an add node or add connection mutation
        :param mutation:
        :param conn: if True checks for add connection mutations, checks for add node mutations otherwise
        :return: true if mutation exists
        """
        if conn:
            return mutation in self.connection_mutations
        else:
            return mutation in self.node_mutations

    def add_mutation(self, mutation: Tuple[int, int], conn: bool):
        if conn:
            if mutation in self.connection_mutations:
                raise Exception('Connection mutation already exists')

            self.connection_mutations[mutation] = self.get_next_connection_id()
            return self._next_conn_id
        else:
            if mutation in self.node_mutations:
                raise Exception('Node mutation already exists')

            self.node_mutations[mutation] = self.get_next_node_id()
            return self._next_node_id

    def get_next_node_id(self) -> int:
        self._next_node_id += 1
        return self._next_node_id

    def get_next_connection_id(self) -> int:
        self._next_conn_id += 1
        return self._next_conn_id
