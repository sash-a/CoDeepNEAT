from typing import Dict, Union

from src2.Genotype.NEAT.Gene import Gene


class MutationRecords:
    def __init__(self, initial_mutations: Dict[Union[int, tuple], int], current_max_node_id, current_max_conn_id):
        """
            Records all mutation in a single run so that no innovation is misused
            simultaneously stores node and connection mutation information
        """

        # maps mutation details to the mutation id
        # for connection the mapping is from (from_node_id, to_node_id) to -> connection_mutation_id
        self.mutations = initial_mutations

        self._next_node_id = current_max_node_id
        self._next_conn_id = current_max_conn_id

    def __repr__(self):
        return repr(self.mutations)

    def exists(self, mutation) -> bool:
        # print("checking existence of mut",mutation,"in",self.mutations, "result",(mutation in self.mutations.keys()))
        return mutation in self.mutations.keys()

    def add_mutation(self, mutation):
        if type(mutation) == tuple:
            """
                if the mutation key is a tuple of ints, the values are to, from node
                this is the mutation key for a connection
            """
            # Making sure tuple of ints
            for x in mutation:
                if not isinstance(x, int):
                    raise TypeError('Incorrect type passed to mutation: ' + mutation)

            self.mutations[mutation] = self.get_next_connection_id()
            return self._next_conn_id

        elif type(mutation) == int:
            """
                if the mutation key is a single int
            """
            self.mutations[mutation] = self.get_next_node_id()
            return self._next_node_id
        else:
            raise TypeError('Incorrect type passed to mutation: ' + mutation)

    def get_next_node_id(self) -> int:
        self._next_node_id += 1
        return self._next_node_id

    def get_next_connection_id(self) -> int:
        self._next_conn_id += 1
        return self._next_conn_id
