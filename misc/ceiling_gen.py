#ceiling_gen - insert a grid of aruco markers on the ceiling in a Webots sim.
import argparse
import cv2
import numpy as np
from os import path, makedirs

#DEFINES
TAG_TYPE="DICT_4X4_100"
TAG_SIZE=18 #tag edge length in centimeters
ROOM_COORDS=[5.44, 3.66, -16.24, -3.66]
WORLDFILE="../worlds/romer_lab.wbt"
COORDFILE="tagcoords.txt"

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL}

#FUNCS
def gen_tags(tag_type, tag_count):
    '''
    generate the aruco tag images and save under the tags folder.
    generates aruco markers
    also rotates them by 90deg CCW.
    '''
    tag_ids=range(0,tag_count)
    #generate individual tags
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[tag_type])

    for tag_id in tag_ids:
        tag=np.zeros((32, 32,1), dtype='uint8')
        cv2.aruco.drawMarker(arucoDict, tag_id, 32, tag, 1)
        tag=cv2.rotate(tag, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(f"tags//marker_{tag_id}.jpg",tag)


def gen_coords(room_coords, tag_count, grid_size):
    '''
    generate the center coordinates of each tag based on room shape and total tag count.
    room shape assumed to be rectangular.
    '''
    grid_x=grid_size[0]
    grid_y=grid_size[1]
    # room coords is in (x1,y1,x2,y2) format
    x_len=room_coords[2]-room_coords[0]
    y_len=room_coords[3]-room_coords[1]
    # calculate the distance between tags and wslls
    # |------*------*------|
    #   xsep   xsep   xsep
    x_separation=x_len/(grid_x+1)
    y_separation=y_len/(grid_y+1)
    #generate list of x and y coordinates
    tag_coords=[]
    for i in range(0,tag_count):
        tag_coords.append([room_coords[0]+((i//grid_y)+1)*x_separation,
                           room_coords[1]+((i%grid_y)+1)*y_separation])
    return tag_coords

def gen_node(x, y, tagid, size):
    '''
    generate text for a single aruco marker node.
    '''
    res=(
        '    Ceiling {\n'
        f'      translation {x} {y} 3.299\n'
        f'      name "tag{tagid}"\n'
        f'      size {size} {size}\n'
        '      appearance PBRAppearance {\n'
        '        baseColorMap ImageTexture {\n'
        '          url [\n'
        f'            "../misc/tags/marker_{tagid}.jpg"\n'
        '          ]\n'
        '          repeatS FALSE\n'
        '          repeatT FALSE\n'
        '          filtering 0\n'
        '        }\n'
        f'        name "tag{tagid}_texture"\n'
        '      }\n'
        '    }\n')
    return res

def gen_nodes(tag_coords, tag_size):
    '''
    generate the webots nodes for the entire ceiling.
    '''
    #convert size from centimeters to meters
    tag_size=tag_size/100
    nodetext=""
    #generate text for each node
    for tagid, coords in enumerate(tag_coords):
        nodetext+=gen_node(coords[0], coords[1], tagid, tag_size)
    res=(
        'DEF CEILING_TAGS Group {\n'
        '  children [\n'
        f'{nodetext}'
        '  ]\n'
        '}\n')
    return res

def add_to_map(text, worldfile, delete):
    '''
    add the generated text to the world file.
    '''
    if not delete:
        with open(worldfile, 'a') as f:
            f.write(text)
    else:
        with open(worldfile, 'r+') as f:
            ftext=f.read()
            ind=ftext.index("DEF CEILING_TAGS Group {")
            f.seek(0)
            f.write(ftext[0:ind])
            f.write(text)
            f.truncate()

def write_coords(text, coordfile):
    '''
    write generated coords to a text file.
    '''
    with open(coordfile, 'w') as f:
        f.write(str(text))

if __name__=="__main__":
    #cli stuff
    parser=argparse.ArgumentParser(
        prog="ceiling_gen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Insert a grid of aruco markers on the ceiling in a Webots sim.",
        epilog='''If called with the -p flag, nothing is written to the files and the outputs are printed to stdout. Tag layout is as follows:
    -y
  ┌─────►
  │
-x│ ┌──────────────────────────────────────────────────────────────────┐
  │ │               ▲                                                  │
  ▼ │               │                                                  │
    │               │xsep                                              │
    │               │                                                  │
    │               ▼                                                  │
    │               0                1               2                 │
    │◄─────────────►▲◄──────────────►                                  │
    │       ysep    │        ysep                                     ┌┼┐
    │               │xsep                                             │┼│
    │               │                                                 │┼│door
    │               ▼                                                 │┼│
    │               3                4               5                │┼│
    │                                                                 └┼┘
    │                                                                  │
''')
    
    parser.add_argument('count', type=int, help="Number of tags to place on the ceiling.")
    
    parser.add_argument('grid_size', type=lambda s: [int(item) for item in s.split(' ')], help="The row-column counts of the grid to lay the tags on. Format is \"x y\"")
    parser.add_argument('--room', type=lambda s: [float(item) for item in s.split(' ')], default=ROOM_COORDS, help=f"The room's corner points. Format is \"x1 y1 x2 y2\". Defaults to {ROOM_COORDS}.")
    parser.add_argument('--type', choices=ARUCO_DICT, default=TAG_TYPE, help=f"Marker type to use. Defaults to {TAG_TYPE}.")
    parser.add_argument('--size', type=float, default=TAG_SIZE, help=f"Length of an edge of the marker, in centimeters. Defaults to {TAG_SIZE}.")
    parser.add_argument('--map', metavar="WBT FILE", default=WORLDFILE, help=f"Webots world file to add the nodes to. This does not delete any previous marker nodes in the world by default. Using -d will delete previous markers, but use only if the marker group is the last node in your world. Defaults to {WORLDFILE}.")
    parser.add_argument('--out', metavar="OUTPUT FILE", default=COORDFILE, help=f"File to write tag coordinates to. Defaults to {COORDFILE}.")
    parser.add_argument('-p', action='store_true', help="Don't save to file, print output.")
    parser.add_argument('-d', action='store_true', help="Delete the previous CEILING_TAGS node in the world map. USE ONLY IF THE CEILING_TAGS NODE IS THE LAST NODE IN THE WORLD MAP.")

    args=parser.parse_args()
    
    
    #create tags folder if it doesn't exist
    if not path.exists("./tags"):
        makedirs("./tags")
    #regenerates tag images on each run, a bit inefficient but caching these correctly is bothersome
    gen_tags(args.type, args.count)
    #calculate tag coordinates
    tag_coords=gen_coords(args.room, args.count, args.grid_size)
    #generate webots nodes
    text=gen_nodes(tag_coords, args.size)
    # display/write webots nodes and tag coords
    if not args.p:
        add_to_map(text, args.map, args.d)
        write_coords(tag_coords, args.out)
    else:
        print(f"add to .wbt file:\n{text}")
        print(f"tag coordinates:\n{tag_coords}")
