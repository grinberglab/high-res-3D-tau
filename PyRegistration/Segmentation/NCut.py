from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from skimage import measure as meas
from numpy import nonzero
import scipy.spatial
from skimage import draw
import matplotlib
import GraphCut as gc

cmap = matplotlib.cm.get_cmap('Reds')

def vector_distance(v1,v2):
    d = np.linalg.norm(v1-v2)
    return d

def display_edges(image, g, threshold=0.01):
    #W = [graph[a][b]['weight'] for a,b in graph.edges()]
    image = image.copy()
    for edge in g.edges():
        n1, n2 = edge
        weight = g[n1][n2]['weight']
        color = cmap(weight)
        color = np.array(color)
        r1,c1 = map(int, g.node[n1]['centroid'])
        r2,c2 = map(int, g.node[n2]['centroid'])
        # if weight >= threshold:
        #     liner,linec = draw.line(r1, c1, r2, c2)
        #     image[liner,linec,...] = color[0:3]*255
        liner, linec = draw.line(r1, c1, r2, c2)
        image[liner, linec, ...] = color[0:3] * 255
    for node in g.nodes():
        n = node
        r1, c1 = map(int, g.node[n]['centroid'])
        circler, circlec = draw.circle(r1, c1, 5)
        label = g.node[n]['label']
        image[circler, circlec, ...] = [label,100,10]

    return image


def compute_distances(G):
    for ed in G.edges():  # the index in the points array in the same number used in the labels matrix
        n1 = ed[0]
        n2 = ed[1]
        c1 = G.node[n1]['mean']
        c2 = G.node[n2]['mean']
        v1 = G.node[n1]['centroid']
        v2 = G.node[n2]['centroid']
        w1 = vector_distance(c1,c2)
        w2 = vector_distance(v1,v2)
        G[n1][n2]['w1'] = w1  # add weight attribute to edge
        G[n1][n2]['w2'] = w2  # add weight attribute to edge

def compute_weights(graph,sigma1=0.2,sigma2=0.2):
    compute_distances(graph)
    #normalize distances to [0,1] interval and compute similarity metric
    W1 = [graph[a][b]['w1'] for a,b in graph.edges()]
    minW1 = min(W1)
    maxW1 = max(W1)
    W2 = [graph[a][b]['w2'] for a,b in graph.edges()]
    minW2 = min(W2)
    maxW2 = max(W2)
    for ec in graph.edges():
        n1,n2 = ec
        w1 = graph[n1][n2]['w1']
        w2 = graph[n1][n2]['w2']
        w1n = (w1 - minW1)/(maxW1 - minW1)
        w2n = (w2 - minW2)/(maxW2 - minW2)
        graph[n1][n2]['w1'] = w1n
        graph[n1][n2]['w2'] = w2n
        w11 = np.math.e ** (-(w1n ** 2) / sigma1) #color distance
        w22 = np.math.e ** (-(w2n ** 2) / sigma2) #centroid distance
        w = w11*w22
        graph[n1][n2]['weight'] = w

def compute_ncut(G,thresh=0.4,nncut=10):
    for node in G.nodes(): # set edges to self to the maximum value
        G.add_edge(node, node, weight=1.0)
    gc._ncut_relabel(G, thresh, nncut)


def run_ncut(img_dic):
    img = img_dic['img']
    mask = img_dic['mask']
    L = img_dic['L']
    A = img_dic['A']
    B = img_dic['B']
    img_center = img_dic['img_center'] # (row,col)
    int_dist = img_dic['dist'] # distance between img center and obj of interest center
    R = img[...,0]
    G = img[...,1]
    BB = img[...,2]

    labels = meas.label(mask)
    props = meas.regionprops(labels)
    nL = len(props)

    if nL < 4:
        idx_brain = nonzero(mask == False)
        img[idx_brain[0], idx_brain[1], 0] = 0
        img[idx_brain[0], idx_brain[1], 1] = 0
        img[idx_brain[0], idx_brain[1], 2] = 0

        return img, mask

    points = []
    for i in range(nL):
        c = props[i].centroid
        c = (c[0],c[1]) #r,c format
        points.append(c)

    # make a Delaunay triangulation of the point data
    delTri = scipy.spatial.Delaunay(points)

    # create a set for edges that are indexes of the points
    edges = set()
    for n in xrange(delTri.nsimplex):
        edge = sorted([delTri.vertices[n,0], delTri.vertices[n,1]])
        edges.add((edge[0], edge[1]))
        edge = sorted([delTri.vertices[n,0], delTri.vertices[n,2]])
        edges.add((edge[0], edge[1]))
        edge = sorted([delTri.vertices[n,1], delTri.vertices[n,2]])
        edges.add((edge[0], edge[1]))
    graph = nx.Graph(list(edges))

    #add the centroid and mean color info to each node
    ctr_dist = -1
    main_node = -1

    # plt.imshow(mask)
    # plt.show()

    closest = -1
    for n in range(nL): #the index in the points array in the same number used in the labels matrix
        idxR,idxC = nonzero(labels == n+1)
        pixL = L[idxR,idxC]
        pixA = A[idxR,idxC]
        pixB = B[idxR,idxC]
        mL = np.mean(pixL)
        mA = np.mean(pixA)
        mB = np.mean(pixB)

        pixR = R[idxR,idxC]
        pixG = G[idxR,idxC]
        pixBB = BB[idxR,idxC]
        mR = np.mean(pixR)
        mG = np.mean(pixG)
        mBB = np.mean(pixBB)

        col = points[n][1]
        row = points[n][0]
        graph.node[n]['centroid'] = np.array([row,col])
        graph.node[n]['mean'] = np.array([mL,mA,mB])
        graph.node[n]['mean_rgb'] = np.array([mR,mG,mBB])
        graph.node[n]['label'] = 0
        graph.node[n]['mask_label'] = n+1

        # plt.plot(col,row,'go',markersize=10)

        dist = np.sqrt((img_center[0]-row)**2 + (img_center[1]-col)**2) #distance from node obj to img center
        if closest == -1 or dist < closest:
            closest = dist
            main_node = n

        # if np.fabs(dist-int_dist) < 0.001: # float comparison :)
        #     plt.plot(col, row, 'mo', markersize=10)
        #     main_node = n

    # col = points[main_node][1]
    # row = points[main_node][0]
    # plt.plot(col,row,'mo',markersize=10)


    # compute edge weights
    compute_weights(graph,1.5,1.5)

    img_g = display_edges(img,graph)
    plt.imshow(img_g)

    # trim graph to leave only nodes that are connected to the main object
    node_dic = graph[main_node]
    nodes = node_dic.keys()
    nodes.insert(0,main_node)
    nodes.sort()
    graph2 = graph.subgraph(nodes).copy()

    img_g = display_edges(img,graph2)
    plt.imshow(img_g)

    # compute ncut
    compute_ncut(graph2,thresh=0.4,nncut=10)

    # img_g = display_edges(img,graph2)
    # plt.imshow(img_g)

    # get structures of interest
    ncut_label = graph2.node[main_node]['label']
    # nodes that have structure of interest label
    brain_nodes = []
    brain_labels = []
    for nn in graph2.nodes():
        if graph2.node[nn]['label'] == ncut_label:
            brain_nodes.append(nn)
            brain_labels.append(graph.node[nn]['mask_label'])

    mask_final = np.zeros(mask.shape)
    for bl in brain_labels:
        idx2 = nonzero(labels == bl)
        mask_final[idx2[0],idx2[1]] = True

    #plt.imshow(mask_final)

    idx_brain = nonzero(mask_final == False)
    img[idx_brain[0],idx_brain[1],0] = 0
    img[idx_brain[0],idx_brain[1],1] = 0
    img[idx_brain[0],idx_brain[1],2] = 0

    return img,mask_final


