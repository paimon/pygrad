from graphviz import Digraph


def top_sort(root):
    result = []
    visited = set()

    def add(val):
        if val in visited:
            return
        visited.add(val)
        for p in val.prev:
            add(p)
        result.append(val)

    add(root)
    return result


def draw_graph(root, rankdir='LR'):
    graph = Digraph(format='svg', graph_attr=dict(rankdir=rankdir))
    visited = set()

    def build(val):
        if val in visited:
            return
        visited.add(val)
        val_id = str(id(val))
        graph.node(val_id, f'{val.operation}|{val.data}|{val.grad}', shape='record')
        for p in val.prev:
            build(p)
            graph.edge(str(id(p)), val_id)

    build(root)
    return graph
