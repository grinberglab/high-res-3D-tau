from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from skimage import measure as meas
from numpy import nonzero
import scipy.spatial
from skimage import draw
import matplotlib
import reg_graph_cut as gc

cmap = matplotlib.cm.get_cmap('Reds')

def edge_similarity(c1,c2,sigma=5):
    # c1 = (L1,A1,B1)
    # c2 = (L1,A1,B1)
    d = np.linalg.norm(c1-c2) #vector distance
    w = np.math.e ** (-(d ** 2) / sigma)
    return w

def edge_similarity2(c1,c2,v1,v2,r=0,sigma1=0.2, sigma2=0.2):
    # c1 = (L1,A1,B1), c2 = (L1,A1,B1): color vectors
    # v1 = (x1,y1), v2 = (x2,y2): centroid coordinates
    # r: distance threshold
    # sigma
    # dc = np.linalg.norm(c1-c2) # vector distance
    # dsp = np.linalg.norm(v1-v2) # centroid distance

    v1 = np.array([v1[1],v1[0]])
    v2 = np.array([v2[1],v2[0]])

    dc = 0.5*(np.var(c1-c2)/(np.var(c1)+np.var(c2))) #normalized euclidean distance is in the interval [0,1]
    dsp = 0.5*(np.var(v1-v2)/(np.var(v1)+np.var(v2)))

    w1 = np.math.e ** (-(dc ** 2) / sigma1)
    w2 = np.math.e ** (-(dsp ** 2) / sigma2)

    w = dsp # the further away, the less desirable the structure is
    return w

def vector_distance(v1,v2):
    d = np.linalg.norm(v1-v2)
    return d

def display_edges(image, g, threshold=0.01):
    #W = [graph[a][b]['weight'] for a,b in graph.edges()]
    image = image.copy()
    for edge in g.edges_iter():
        n1, n2 = edge
        weight = g[n1][n2]['weight']
        color = cmap(weight)
        color = np.array(color)
        c1,r1 = map(int, g.node[n1]['centroid'])
        c2,r2 = map(int, g.node[n2]['centroid'])
        # if weight >= threshold:
        #     liner,linec = draw.line(r1, c1, r2, c2)
        #     image[liner,linec,...] = color[0:3]*255
        liner, linec = draw.line(r1, c1, r2, c2)
        image[liner, linec, ...] = color[0:3] * 255
    for node in g.nodes_iter():
        n = node
        c1, r1 = map(int, g.node[n]['centroid'])
        circler, circlec = draw.circle(r1, c1, 10)
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
        w11 = np.math.e ** (-(w1n ** 2) / sigma1)
        w22 = np.math.e ** (-(w2n ** 2) / sigma2)
        w = w11*w22
        graph[n1][n2]['weight'] = w

def compute_ncut(G,thresh=0.4,nncut=2):
    for node in G.nodes_iter(): # set edges to self to the maximum value
        G.add_edge(node, node, weight=1.0)
    gc._ncut_relabel(G, thresh, nncut)

# BEGIN #
struct = np.load('/home/maryana/Projects/workspace-python/PyRegistration/struct2.npy').item()
img = struct['img']
mask = struct['mask']
L = struct['L']
A = struct['A']
B = struct['B']
img_center = struct['img_center'] # (row,col)
int_dist = struct['dist'] # distance between img center and obj of interest center

R = img[...,0]
G = img[...,1]
BB = img[...,2]

labels = meas.label(mask)
props = meas.regionprops(labels)
nL = len(props)
points = []
for i in range(nL):
    c = props[i].centroid
    c = (c[1],c[0]) #r,c format
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
for n in range(nL): #the index in the points array in the same number used in the labels matrix
    idx = nonzero(labels == n+1)[0]
    pixL = L[idx]
    pixA = A[idx]
    pixB = B[idx]
    mL = np.mean(pixL)
    mA = np.mean(pixA)
    mB = np.mean(pixB)

    pixR = R[idx]
    pixG = G[idx]
    pixBB = BB[idx]
    mR = np.mean(pixR)
    mG = np.mean(pixG)
    mBB = np.mean(pixBB)

    row = points[n][0]
    col = points[n][1]
    graph.node[n]['centroid'] = np.array([row,col])
    graph.node[n]['mean'] = np.array([mL,mA,mB])
    graph.node[n]['mean_rgb'] = np.array([mR,mG,mBB])
    graph.node[n]['label'] = 0
    graph.node[n]['mask_label'] = n+1

    dist = np.sqrt((img_center[1]-row)**2 + (img_center[0]-col)**2) #distance from node obj to img center
    if np.fabs(dist-int_dist) < 0.001: # float comparison :)
        main_node = n

compute_weights(graph,0.2,0.2)

for ee in graph.edges():
    n1,n2 = ee
    c1 = graph.node[n1]['centroid']
    c2 = graph.node[n2]['centroid']
    d1 = graph[n1][n2]['w1']
    d2 = graph[n1][n2]['w2']
    w1 = np.math.e ** (-(d1 ** 2) / 0.2)
    w2 = np.math.e ** (-(d2 ** 2) / 0.2)
    w = w1*w2
    graph[n1][n2]['weight'] = w

    #d1 = np.linalg.norm(c1-c2)
    #d2 = (0.5*(np.var(c1-c2)/(np.var(c1)+np.var(c2))))
    # p1 = (np.linalg.norm((c1-np.mean(c1)) - (c2-np.mean(c2))))**2
    # p2 = (np.linalg.norm(c1-np.mean(c1))**2) + (np.linalg.norm(c2-np.mean(c2))**2)
    # d3 = 0.5*(p1/p2)
    print(c1,c2,d1,d2,w)


# trim graph to leave only nodes that are connected to the main object
node_dic = graph[main_node]
nodes = node_dic.keys()
nodes.insert(0,main_node)
nodes.sort()
graph2 = graph.subgraph(nodes)

img_g = display_edges(img,graph2)
plt.imshow(img_g)

# compute ncut
compute_ncut(graph2)

img_g = display_edges(img,graph2)
plt.imshow(img_g)

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

plt.imshow(mask_final)

#plot graph
pointIDXY = dict(zip(range(len(points)), points))
plt.imshow(img)
nx.draw(graph, pointIDXY)
plt.show()

