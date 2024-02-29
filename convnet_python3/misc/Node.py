'''
@author: Maryana Alegro

Basic tree node implementation
Used for generating XML metadata
'''


class Node(object):

    def __init__(self,name):
        self.id = name #XML Element name
        self.data = {} #XML Element attributes
        self.children = []
        self.parent = None
        self.xml_elem = None

    def set_xml_elem(self,obj):
        self.xml_elem = obj

    def get_xml_elem(self):
        return self.xml_elem

    def set_parent(self,node):
        self.parent = node

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children

    def add_child(self,node):
        self.children.append(node)
        node.set_parent(self)

    def add_children(self,nodes):
        for n in nodes:
            self.add_child(n)

    def get_node_id(self):
        return self.id

    def get_data(self):
        return self.data

    def set_data(self,dt):
        if type(dt) == dict:
            self.data = dt
        else:
            print('Error: data must be a dictionary.')

    def add_data(self,dt):
        #data must be a dictionary
        if type(dt) == dict:
            self.data.update(dt)
        else:
            print('Error: data must be a dictionary.')
