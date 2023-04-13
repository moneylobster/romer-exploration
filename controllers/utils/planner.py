import pickle
import networkx as nx
import numpy as np
from skimage import morphology, measure
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import sknw
import pickle

def medial_axis(img):
    # invert the image to treat the free space as foreground
    img = img.astype(bool)
    # compute the medial axis
    skel, distance = morphology.medial_axis(img, return_distance=True)
    # create a blank grayscale image to store the medial axis
    med_axis = np.zeros(img.shape, dtype=np.uint8)
    # set the pixel values of the medial axis based on the distance transform
    med_axis[skel] = 255
    # return the medial axis image
    return med_axis

def find_medial_axis_vertices(med_axis):
    # find the contours of the medial axis image
    contours = measure.find_contours(med_axis, 0.5)
    # find the center of each contour
    centers = [np.mean(contour, axis=0).astype(int) for contour in contours]
    vertices = centers
    # merge vertices that are close to each other
    dist_matrix = cdist(vertices, vertices)
    for i, v1 in enumerate(vertices):
        for j, v2 in enumerate(vertices[i+1:], start=i+1):
            if dist_matrix[i, j] < 3:
                vertices[i] = v1
                vertices[j] = v1
    
    return vertices, contours

def cast_rays(img, pixel_coords):    
    # Initialize results list
    results = []    
    # Iterate over pixel coordinates
    for coord in pixel_coords:
        pre_results = []
        # Iterate over angles
        for angle in range(0,360,2):
            # Iterate over distances
            for dist in range(0,100):
                # Calculate x and y coordinates
                x = int(coord[0] + np.cos(np.radians(angle))*dist)
                y = int(coord[1] + np.sin(np.radians(angle))*dist)
                # Check if x and y are within the image
                if x < 0 or x > img.shape[0] - 1 or y < 0 or y > img.shape[1] - 1 or (x,y) in pre_results:
                    break
                # Check if pixel is occupied
                if img[x,y] == 0:
                    pre_results.append((x,y))
                    break
        results.append(pre_results)       
    return results


def is_unique_point(lst,vertex):    
    res = []
    new_lst = []
    res_point = []
    points = set(sum(lst, []))
    print(f"number of detected walls are {len(points)}")    
    sorted_list = sorted(lst, key=lambda x: len(x),reverse=True)
    sorted_index = sorted(range(len(lst)), key=lambda x: len(lst[x]),reverse=True)
    for point in points:
        print(point)
        for i in range(len(sorted_list)):
            if point in sorted_list[i]:
                if point not in res_point:
                    res.append(vertex[sorted_index[i]])
                    for j in range(len(sorted_list[i])):
                        if sorted_list[i][j] not in res_point:
                            res_point.append(sorted_list[i][j])
                break
    print(f"number of detected walls are {len(res_point)}")
    return res, points

def skeleton_image_to_graph(skeIm, connectivity=2):
    assert(len(skeIm.shape) == 2)
    skeImPos = np.stack(np.where(skeIm))
    skeImPosIm = np.zeros_like(skeIm, dtype=np.int)
    skeImPosIm[skeImPos[0], skeImPos[1]] = np.arange(0, skeImPos.shape[1])
    g = nx.Graph()
    if connectivity == 1:
        neigh = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    elif connectivity == 2:
        neigh = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])
    else:
        raise ValueError(f'unsupported connectivity {connectivity}')
    for idx in range(skeImPos[0].shape[0]):
        for neighIdx in range(neigh.shape[0]):
            curNeighPos = skeImPos[:, idx] + neigh[neighIdx]
            if np.any(curNeighPos<0) or np.any(curNeighPos>=skeIm.shape):
                continue
            if skeIm[curNeighPos[0], curNeighPos[1]] > 0:
                g.add_edge(skeImPosIm[skeImPos[0, idx], skeImPos[1, idx]], skeImPosIm[curNeighPos[0], curNeighPos[1]], weight=np.linalg.norm(neigh[neighIdx]))
    g.graph['physicalPos'] = skeImPos.T
    return g

def create_medial_axis(save=False):
    img = cv2.imread("map_framed.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img//240
    img_copy = img.copy()
    # make grayscale image binary

    # compute the medial axis of the free space
    med_axis = medial_axis(img.transpose())
    # find the white pixels in the medial axis image
    white = np.where(med_axis == 255)
    # create a list of vertices from the white pixels where x is white[0] and y is white[1] in numpy array form
    vertices = list(zip(white[0], white[1]))
    
    # construct a networkx graph out of the skeletonized image
    graph=sknw.build_sknw(med_axis)
    # convert the node names into the node coordinates.
    labelmapping={i:tuple(node["o"]) for i, node in graph.nodes(True)}
    nx.relabel_nodes(graph, labelmapping, copy=False)

    if save:
        # save the graph
        with open("roadmap.pickle", "wb") as f:
            pickle.dump(graph, f)

    return graph

class NavChoice():
    
    def __init__(self, current, tovisit, path, dist, choices):
        '''
        current: which point am I on currently?
        tovisit: which points do I need to visit?
        path: how did I get here?
        dist: how much distance did I travel to get here?
        choices: which waypoints did I visit in order to get here?
        '''
        self.current=current
        self.tovisit=tovisit
        self.path=path
        self.dist=dist
        self.choices=choices
        self.to={}
    
    def exploreopts(self, graph, cached):
        '''
        find the path to each waypoint and calculate the cost for each option
        
        graph: the roadmap.
        cached: list of cached astar calcs.
        
        returns a list of the options and their costs, and whether all waypoints have been visited.
        format is [waypoint, cost, done?]
        '''
        
        costs=[]
        
        for waypoint in self.tovisit:
            #print(f"from {self.current} to {waypoint}")
            # check cache - does astar path already exist?
            if str(self.current)+str(waypoint) in cache:
                # we have this exact computation
                cachelabel=str(self.current)+str(waypoint)
                path=cache[cachelabel]["path"]
                dist=cache[cachelabel]["dist"]
            elif str(waypoint)+str(self.current) in cache:
                # we have the inverse of this computation
                cachelabel=str(waypoint)+str(self.current)
                path=cache[cachelabel]["path"]
                # reverse list
                path=[path[i] for i in range(len(path)-1, -1, -1)]
                dist=cache[cachelabel]["dist"]
            else:
                path=nx.astar_path(graph, self.current, waypoint, heuristic=distance)
                
                # convert into floats
                path=[(float(i[0]),float(i[1])) for i in path]
                
                # find how long the path is
                dist=0
                for i in range(len(path)-1):
                    dist+=distance(path[i], path[i+1])

                # save result to cache.
                cachelabel=str(self.current)+str(waypoint)
                cache[cachelabel]={}
                cache[cachelabel]["path"]=path
                cache[cachelabel]["dist"]=dist
            
            wpchoice=NavChoice(waypoint,
                               [i for i in self.tovisit if i != waypoint],
                               self.path+path,
                               self.dist+dist,
                               self.choices+[waypoint])
            
            self.to[str(waypoint)]=wpchoice
            costs.append([wpchoice.choices, wpchoice.dist, len(wpchoice.tovisit)==0])
            
        return costs

def distance(p1,p2):
    '''
    L2 norm
    '''
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
def getpathdict(tree, route):
    '''
    get a node in the tree.
    
    tree: the root of the navigation tree. NavChoice object.
    route: a list with the names of each waypoint to go to.
    '''
    current=tree
    for i in route:
        current=current.to[str(i)]
    return current


def loadgraph():
    # load the roadmap file
    with open("roadmap.pickle", "rb") as f:
        graph=pickle.load(f)
    return graph

def loadwaypoints():
    # load the waypoints file
    with open('foreground_centroids.txt', 'r') as f:
        waypoints = [tuple(map(float, line.split())) for line in f]
    return waypoints

def pathfind(graph, waypoints):
    # add starting point into the waypoints as well as the last element
    startingpoint=(0,0)
    waypoints.append(startingpoint)

    # we need to connect all waypoints to the graph nodes they are closest to.
    nodecoords=list(graph.nodes())

    # store closest node to each one here
    closest=[[waypoint, distance(waypoint,nodecoords[0]), nodecoords[0]] for waypoint in waypoints]

    for node in graph.nodes():
        for j,waypoint in enumerate(closest):
            # if this node is closer to the waypoint, add it as closest
            newdist=distance(waypoint[0], node)
            if newdist < waypoint[1]:
                closest[j]=[waypoint[0], newdist, node]

    # add waypoints to graph. index them as negative.
    for waypoint in closest:
        graph.add_node(waypoint[0])
        graph.add_edge(waypoint[2], waypoint[0])

    # to cache results of each astar operation.
    cache={}

    # do a djikstra search to find the order in which to visit the waypoints
    navtree=NavChoice(startingpoint, waypoints[:-1], [], 0, [])

    toexplore=navtree
    pending=[]
    while True:
        # check all options of the current node to explore
        pending+=toexplore.exploreopts(graph, cache)
        # check if any of the options have completed the task
        completed=[i for i in pending if i[2]]
        if len(completed):
            break
        # sort by distance
        pending.sort(key=lambda x: x[1])
        # cull the last 10% of the list (they probably aren't the right choice anyway)
        # this parameter can be increased to get better results. 100 (cull 1%) finds a shorter path, but takes a few seconds to find it.
        todel=len(pending)//10
        if todel:
            del pending[-todel:]
            #print(f"deleted {todel} elements")
        # get the object for the lowest cost option
        toexplore=getpathdict(navtree, pending.pop(0)[0])

    return completed