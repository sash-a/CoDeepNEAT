from typing import Dict, Tuple


class MutationRecords:
    def __init__(self, initial_connection_mutations: Dict[tuple, int], initial_node_mutations: Dict[tuple, int]):
        """
            Records all mutation in a single run so that no innovation is misused
            simultaneously stores node and connection mutation information
        """

        # maps mutation details to the mutation id
        # for connection the mapping is from (from_node_id, to_node_id) to -> connection_mutation_id
        self.connection_mutations = initial_connection_mutations
        # mapping from (connection id, node_count) -> node id
        # where node_count is the number of times a mutation has occurred on this connection
        self.node_mutations = initial_node_mutations

        self._curr_node_id = 1  # 1 as we start with nodes 0 and 1 as the initialization (they aren't mutated in)
        # -1 as get method increments then returns, meaning 0 will be returned first call and connections are mutated in
        self._curr_conn_id = -1

    def __repr__(self):
        return 'Connections: ' + repr(self.connection_mutations) + '\nnodes: ' + repr(self.node_mutations)

    def connection_mut_exists(self, mutation: Tuple[int, int]):
        """takes in a (from, to) node_id tuple"""
        return mutation in self.connection_mutations

    def node_mut_exists(self, mutation: Tuple[int, int]):
        """takes in a (conn_id, node_count) tuple"""
        return mutation in self.node_mutations

    def add_conn_mutation(self, mutation: Tuple[int, int]):
        if mutation in self.connection_mutations:
            raise Exception(f'Connection mutation already exists. Trying to add connection with from node'
                            f' {mutation[0]} to node {mutation[1]}')

        self.connection_mutations[mutation] = self.get_next_connection_id()
        return self._curr_conn_id

    def add_node_mutation(self, mutation: Tuple[int, int]):
        if mutation in self.node_mutations:
            raise Exception('Node mutation already exists')

        self.node_mutations[mutation] = self.get_next_node_id()
        return self._curr_node_id

    def get_next_node_id(self) -> int:
        self._curr_node_id += 1
        return self._curr_node_id

    def get_next_connection_id(self) -> int:
        self._curr_conn_id += 1
        return self._curr_conn_id
