import pickle
import networkx as nx
import numpy as np
from skimage import morphology, measure
import cv2
from scipy.spatial.distance import cdist
import sknw
import pickle

def medial_axis(img):
    '''
    Form the medial axis image from an image.
    
    img: image to process.
    '''
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


def create_medial_axis(img):
    '''
    Create a medial axis graph from an image.
    
    img: image to process.
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img//240

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

    return img, vertices, graph

def count_neighbors(point, medial):
    '''
    Returns a list of valid neighbors for the given point on the grid map
    '''
    x, y = point
    neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1), (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)]
    valid_neighbors = []
    for neighbor in neighbors:
        if neighbor in medial:
            valid_neighbors.append(neighbor)
    return len(valid_neighbors)

def find_waypoints(img, vertices):
    '''
    Find the waypoints to go to that provide good coverage of all the
    walls in the map. Visiting all these waypoints should ideally mean the robot
    has looked at every part of the map walls.
    
    img: Occupancy grid image
    vertices: medial axis points, given as a list.
    '''
    
    intersection = []
    for point in vertices:
        num_neighbors = count_neighbors(point, vertices)
        if num_neighbors >= 3:
            intersection.append(point)

    # create a black image
    img2 = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    # make intersections white
    for point in intersection:
        img2[int(point[1]), int(point[0])] = (255, 255, 255)

    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    # Define the kernel size for erosion and dilation
    kernel_size = (23, 23)

    # Create the erosion kernel
    erosion_kernel = np.ones(kernel_size, np.uint8)

    # Erode the image
    dilated_img = cv2.dilate(gray, erosion_kernel, iterations=1)

    # erode the image
    #eroded_img = cv2.erode(dilated_img, erosion_kernel, iterations=1)

    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_img, connectivity=8)

    centroids = centroids[1:]
    # Return the list of foreground centroids pixel coordinates
    return centroids

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
    
    def exploreopts(self, graph, cache):
        '''
        find the path to each waypoint and calculate the cost for each option
        
        graph: the roadmap.
        cache: list of cached astar calcs.
        
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
    
    p1: 2D vector
    p2: 2D vector
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

def pathfind(graph, waypoints, cull_percentage):
    '''
    Find a path through the graph that visits each waypoint, travelling the shortest distance.
    
    graph: networkx graph of roadmap.
    waypoints: the waypoints to visit as a 2d list.
    cull_percentage: which percentage of the worst plans so far to cull on each iteration.
    
    returns:
        stats: a list in the format: [[waypoint visit order, total distance, True]]
        path: a list of 2d coordinates to go to in order to enact the motion plan.
    '''
    waypoints=[tuple(i) for i in waypoints]
    
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
        # cull the last cull_percentage% of the list (they probably aren't the right choice anyway)
        # this parameter can be increased to get better results. 100 (cull 1%) finds a shorter path, but takes a few seconds to find it.
        todel=len(pending)//(100//cull_percentage)
        if todel:
            del pending[-todel:]
            #print(f"deleted {todel} elements")
        # get the object for the lowest cost option
        toexplore=getpathdict(navtree, pending.pop(0)[0])

    return completed, getpathdict(navtree, completed[0][0]).path

def pathplan(imgpath="map_framed.png", cull_percentage=10):
    '''
    Create a path plan from the occupancy grid.
    
    imgpath: path to occupancy grid map of current world.
    cull_percentage: which percentage of the worst plans so far to cull on each iteration.
    
    returns:
        stats: a list in the format: [[waypoint visit order, total distance, True]]
        path: a list of 2d coordinates to go to in order to enact the motion plan.
    '''
    # load occupancy grid map
    img = cv2.imread(imgpath)
    # create medial axis graph
    img, vertices, graph = create_medial_axis(img)
    # find the waypoints to go to
    waypoints=find_waypoints(img, vertices)
    # find how best to go to the waypoints
    stats, path=pathfind(graph, waypoints, cull_percentage)
    return stats, path