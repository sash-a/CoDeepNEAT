import matplotlib.pyplot as plt
import math


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
        self.traversalID = ""# input to output

    def add_child(self, value=None):
        self.add_child(Node(value))

    def add_child(self, child_node, connection_type_is_module = True):
        """
        :param childNode: Node to be added - can have subtree underneath
        """
        self.children.append(child_node)
        child_node.parents.append(self)

    def remove_child(self, child):
        if(not child in self.children):
            print("cannot remove child",child," from",self, " it is not present in children list")
            return
        self.children.remove(child)
        child.parents.remove(self)

    def get_child(self, childNum):
        return self.children[childNum]

    def get_output_node(self):
        if (len(self.children) == 0):
            return self

        return self.children[0].get_output_node()

    def get_input_node(self):
        if len(self.parents) == 0:
            return self

        return self.parents[0].get_input_node()

    def get_traversal_ids(self, current_id=""):
        """should be called on root node
            calculates all nodes traversal ID
        """
        self.traversalID = current_id
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

    def print_tree(self, nodes_printed=None):
        if nodes_printed is None:
            nodes_printed = set()

        if self in nodes_printed:
            return

        nodes_printed.add(self)
        self.print_node()

        for child in self.children:
            child.print_tree(nodes_printed)

    def print_node(self, print_to_console=True):
        pass

    def plot_tree(self, nodes_plotted=None, rot_degree=0, title = ""):
        if nodes_plotted is None:
            nodes_plotted = set()

        arrow_scale_factor = 1

        y = len(self.traversalID)
        x = 0

        count = 0
        for id in self.traversalID.split(","):
            if(id == "_"):
                continue
            y+= int(id)*count
            count+=1


        for i in range(4):
            x += self.traversalID.count(repr(i)) * i

        # x +=y*0.05

        x = x * math.cos(rot_degree) - y * math.sin(rot_degree)
        y = y * math.cos(rot_degree) + x * math.sin(rot_degree)

        if self in nodes_plotted:
            return x, y

        nodes_plotted.add(self)

        plt.plot(x, y, self.get_plot_colour(), markersize=10)

        for child in self.children:
            c = child.plot_tree(nodes_plotted, rot_degree)
            if c is not None:
                cx, cy = c
                plt.arrow(x, y, (cx - x) * arrow_scale_factor, (cy - y) * 0.8 * arrow_scale_factor, head_width=0.13,
                          length_includes_head=True)

        if self.is_input_node():
            plt.title(title)
            plt.show()

        return x, y

    def get_plot_colour(self):
        return 'ro'


def gen_node_graph(node_type, graph_type="diamond", linear_count=1):
    """the basic starting points of both blueprints and modules"""
    #print("initialising graph",node_type,"of shape",graph_type)
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
