
from misc.Tree import Tree
from misc.Node import Node


def main():

    # a = Node('Tiles')
    # a.add_data({'total_rows':100, 'total_cols':200, 'home':'/a/b/c'})
    #
    # b = Node('Tile 1')
    # b.add_data({'rows': 10, 'cols': 20, 'file': 'tile1.tif'})
    # bN = Node('Noth 1')
    # bS = Node('South 1')
    # bW = Node('West 1')
    # bE = Node('East 1')
    # b.add_child(bN)
    # b.add_child(bS)
    # b.add_child(bW)
    # b.add_child(bE)

    a = Node('a')
    a.add_data({'a1':'1'})

    b = Node('b')
    b.add_data({'b2': '2'})
    g = Node('g')
    h = Node('h')
    a.add_children([b,g,h])

    c = Node('c')
    d = Node('d')
    b.add_children([c,d])

    e = Node('e')
    f = Node('f')
    c.add_children([e,f])

    i = Node('i')
    h.add_children([i])

    j = Node('j')
    i.add_children([j])

    tree = Tree(a)
    xml_str = tree.export_xml_string()
    print(xml_str)




if __name__ == '__main__':
    print('main')
    main()