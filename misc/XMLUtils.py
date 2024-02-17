
from misc.Node import Node
from misc.Tree import Tree
from lxml import etree as ET


class XMLUtils(object):

    @staticmethod
    def parse_dict(dics,parent=None):
        # dics is always and array with 1 or more dictionaries. The head must always have 1 single dict.

        node = None

        nD = len(dics)
        for d in range(nD):
            dic = dics[d] #get obj from array
            node = Node(dic['name'])

            #print(dic['name'])

            if not parent is None:  # add curr node to parents children list
                parent.add_child(node)

            node.set_parent(parent)
            node_dic = {}

            att = dic['attrib'] #another dict
            for key in att.keys():
                if key == 'children':
                    arr_dic = att[key]
                    XMLUtils.parse_dict(arr_dic,node)
                else:
                    node_dic[key] = att[key] #it's a leave node
                    node.add_data(node_dic)

        return node

    @staticmethod
    def parse_tiles_metadata(xml_file):
        xml_tree = ET.parse(xml_file)

        image_node = xml_tree.xpath('//Image')[0]
        img_rows = int(image_node.attrib['rows'])
        img_cols = int(image_node.attrib['cols'])
        img_home = str(image_node.attrib['home'])
        img_file = str(image_node.attrib['file'])

        tiles_node = xml_tree.xpath('//Tiles')[0]
        grid_cols = int(tiles_node.attrib['grid_cols'])
        grid_rows = int(tiles_node.attrib['grid_rows'])

        return grid_rows, grid_cols, img_rows, img_cols, img_home, img_file




    @staticmethod
    def dict2xml(dic):
        if type(dic) != list:
            dic = [dic]
        head = XMLUtils.parse_dict(dic)
        tree = Tree(head)
        return tree.export_xml_string()

    @staticmethod
    def dict2xmlfile(dic,xml_file):
        if type(dic) != list:
            dic = [dic]
        head = XMLUtils.parse_dict(dic)
        tree = Tree(head)
        with open(xml_file, 'wb+') as out:
            out.write(tree.export_xml_string())



def main():

    e = dict()
    e['name'] = 'e'
    e['attrib'] = {}

    f = dict()
    f['name'] = 'f'
    f['attrib'] = {'etc':'f'}

    c = dict()
    c['name'] = 'c'
    c['attrib'] = {'children':[e,f], 'att':'123', 'att2':'123'}

    d = dict()
    d['name'] = 'd'
    d['attrib'] = {}

    b = dict()
    b['name'] = 'b'
    b['attrib'] = {'children':[c,d]}

    a = dict()
    a['name'] = 'a'
    a['attrib'] = {'children':[b]}

    dics = [a]

    #xmlUtils = XMLUtils()
    str_xml = XMLUtils.dict2xml(dics)
    print(str_xml)



if __name__ == '__main__':
    main()