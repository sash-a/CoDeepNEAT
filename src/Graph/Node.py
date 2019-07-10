import matplotlib.pyplot as plt
import math
import graphviz


class Node:
    """
    All children lead to the leaf node
    """

    children = []
    parents = []
    species_number = None

    # a string structured as '1,1,3,2,0' where each number represents which child to move to along the path from

    def __init__(self, val=None):
        self.species_number = val
        self.children = []
        self.parents = []
        self.traversal_id = ""  # input to output

    def add_child(self, value=None):
        self.add_child(Node(value))

    def add_child(self, child_node):
        """
        :param childNode: Node to be added - can have subtree underneath
        """
        if child_node in self.children:
            raise Exception("node", child_node, "already childed to", self)
        self.children.append(child_node)
        child_node.parents.append(self)

    def get_child(self, childNum):
        return self.children[childNum]

    def get_output_node(self):
        if self.is_output_node():
            return self

        return self.children[0].get_output_node()

    def get_input_node(self):
        if self.is_input_node():
            return self

        return self.parents[0].get_input_node()

    def get_traversal_ids(self, current_id=""):
        """should be called on root node
            calculates all nodes traversal ID
        """

        if not self.traversal_id == "":
            return

        self.traversal_id = current_id
        # print(self,"num children:", len(self.children))
        # print("Me:",self,"child:",self.children[0])
        for childNo in range(len(self.children)):
            new_id = current_id + (',' if not current_id == "" else "") + repr(childNo)
            # print(newID)
            self.children[childNo].get_traversal_ids(new_id)

    def is_input_node(self):
        return len(self.parents) == 0

    def is_output_node(self):
        return len(self.children) == 0

    def has_siblings(self):
        for parent in self.parents:
            if len(parent.children) > 1:
                return True

        return False

    def plot_tree_with_graphvis(self , title="", graph = None,nodes_plotted=None, file = "temp"):
        file = "../Analysis/Graphs/"+file

        if graph is None:
            graph = graphviz.Digraph(comment=title)

        if nodes_plotted is None:
            nodes_plotted = set()
        else:
            if self in nodes_plotted:
                # print("node",self.traversal_id,self.get_layer_type_name()," already plotted")
                return

        nodes_plotted.add(self)
        # print('adding node',self.traversal_id,self.get_layer_type_name()    )

        prefix = 'INPUT\n' if self.is_input_node() else ("OUTPUT\n" if self.is_output_node() else '')
        # print("pref:",prefix,"id:",self.traversal_id)
        graph.node(self.traversal_id, (prefix + self.get_layer_type_name()), style="filled",
                   fillcolor=self.get_plot_colour(include_shape=False))
        for child in self.children:
            child.plot_tree_with_graphvis(graph=graph, nodes_plotted=nodes_plotted)
            graph.edge(self.traversal_id, child.traversal_id)

        if self.is_input_node():
            graph.render(file, view=True)

    def get_layer_type_name(self):
        raise Exception("override layer type name is super classes")

    def plot_tree_with_matplotlib(self, nodes_plotted=None, rot_degree=0, title=""):
        if nodes_plotted is None:
            nodes_plotted = set()

        arrow_scale_factor = 1

        y = len(self.traversal_id)
        x = 0

        count = 0
        for id in self.traversal_id.split(","):
            if (id == "_"):
                continue
            y += int(id) * count
            count += 1

        for i in range(4):
            x += self.traversal_id.count(repr(i)) * i

        # x +=y*0.05

        x = x * math.cos(rot_degree) - y * math.sin(rot_degree)
        y = y * math.cos(rot_degree) + x * math.sin(rot_degree)

        if self in nodes_plotted:
            return x, y

        nodes_plotted.add(self)

        plt.plot(x, y, self.get_plot_colour(), markersize=10)

        for child in self.children:
            c = child.plot_tree_with_matplotlib(nodes_plotted, rot_degree)
            if c is not None:
                cx, cy = c
                plt.arrow(x, y, (cx - x) * arrow_scale_factor, (cy - y) * 0.8 * arrow_scale_factor, head_width=0.13,
                          length_includes_head=True)

        if self.is_input_node():
            plt.title(title)
            plt.show()

        return x, y

    def clear(self):

        for node in self.get_all_nodes_via_bottom_up(set()):
            node.traversal_id = ""

    def get_all_nodes_via_bottom_up(self, nodes: set):
        if (self in nodes):
            return

        nodes.add(self)
        for child in self.children:
            child.get_all_nodes_via_bottom_up(nodes)

        return nodes

    def get_all_nodes_via_top_down(self, nodes: set):
        """should be called from the output node"""
        if (self in nodes):
            return

        nodes.add(self)
        for parent in self.parents:
            parent.get_all_nodes_via_top_down(nodes)

        return nodes

    def severe_node(self):
        """removes this node entirely from the graph.
        removes self as a child of all parents and removes self as a parent of all children"""
        # print("severing node",self)
        for parent in self.parents:
            # print("severing",self, "from parent",parent)
            parent.children.remove(self)

        for child in self.children:
            # print("severing",self, "from parent",child)
            child.parents.remove(self)


def gen_node_graph(node_type, graph_type="diamond", linear_count=1):
    """the basic starting points of both blueprints and modules"""
    # print("initialising graph",node_type,"of shape",graph_type)
    input = node_type()

    if graph_type == "linear":
        head = input
        for i in range(linear_count - 1):
            head.add_child(node_type())
            head = head.children[0]

    if graph_type == "diamond":
        head = input
        for i in range(linear_count):
            head.add_child(node_type())
            head.add_child(node_type())
            head.children[0].add_child(node_type())
            head.children[1].add_child(head.children[0].children[0])
            head = head.children[0].children[0]

    if graph_type == "triangle":
        """feeds input node to a child and straight to output node"""
        head = input
        for i in range(linear_count):
            head.add_child(node_type())
            head.children[0].add_child(node_type())
            head.add_child(head.children[0].children[0])
            head = head.children[0].children[0]

    if graph_type == "single":
        pass

    return input
