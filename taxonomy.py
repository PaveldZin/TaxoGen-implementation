from anytree import NodeMixin, RenderTree

class TaxonomyNode(NodeMixin):
    def __init__(self, name=None, parent=None, children=None, data=None):
        super(TaxonomyNode, self).__init__()
        self.name = name
        self.parent = parent
        if children:
            self.children = children
        self.data = data

    def print_tree(self):
        for pre, fill, node in RenderTree(self):
            treestr = u"%s%s" % (pre, node.name)
            if node.name == 'root':
                print(treestr.ljust(8))
            else:
                print(treestr.ljust(8), node.data.top_terms())