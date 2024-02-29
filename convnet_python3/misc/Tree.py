
'''
@author: Maryana Alegro

Basic tree implementation
Used for generating XML metadata
'''

from lxml import etree as ET
from collections import deque
#import xmljson
import json


class Tree(object):

    def __init__(self,h):
        self.head = h #Head is supposed to be Node type


    #traverse tree and create XML Tree object
    def export_xml_tree(self):
        #create root XML node
        visited = deque()
        root_xml = ET.Element(self.head.get_node_id(), attrib=self.head.get_data())
        self.head.set_xml_elem(root_xml)

        # create XML using  BFS
        #prev_xml = None #pointer to XML nodes
        visited.appendleft(self.head) #nodes queue
        while(visited): # loop while queue is not empty
            curr = visited.pop()
            if not curr.get_parent() is None: # it not head, get xml elem pointer from parent
                curr_xml = ET.SubElement(curr.get_parent().get_xml_elem(), curr.get_node_id(), attrib=curr.get_data())
                curr.set_xml_elem(curr_xml)

            # add node's children to queue
            for node in curr.get_children():
                visited.appendleft(node)

        xml_tree = ET.ElementTree(root_xml)
        return xml_tree

    #export XML as formatted string
    def export_xml_string(self):
        xml_tree = self.export_xml_tree()
        return ET.tostring(xml_tree, pretty_print=True, xml_declaration=True, encoding='UTF-8')


    # def export_json(self):
    #     xml_tree = self.export_xml_tree()
    #     js = json.dumps(xmljson.badgerfish.data(xml_tree))
    #     return js




