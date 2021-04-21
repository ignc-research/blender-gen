import random

def orderCorners(objBB):
    """change bounding box corner order."""
    # change bounding box order according to
    # https://github.com/F2Wang/ObjectDatasetTools/blob/master/create_label_files.py
    out = []
    corners = [v[:] for v in objBB]  # list of tuples (x,y,z)
    out.append(corners[0])  # -1 -1 -1
    out.append(corners[1])  # -1 -1 1
    out.append(corners[3])  # -1 1 -1
    out.append(corners[2])  # -1 1 1
    out.append(corners[4])  # 1 -1 -1
    out.append(corners[5])  # 1 -1 1
    out.append(corners[7])  # 1 1 -1
    out.append(corners[6])  # 1 1 1
    return out

def kelvin_to_rgb(K):
    """converts color temperature in Kelvin to RGB values according to
    http://www.vendian.org/mncharity/dir3/blackbody/UnstableURLs/bbr_color.html"""
    table = {4000: (1.0000, 0.6636, 0.3583),
             5000: (1.0000, 0.7992, 0.6045),
             6000: (1.0000, 0.9019, 0.8473),
             7000: (0.9337, 0.9150, 1.0000),
             8000: (0.7874, 0.8187, 1.0000),
             9000: (0.6693, 0.7541, 1.0000),
             0:    (1,1,1)
            }
    rgb = table[K]
    return rgb

def get_random_temperature_color():
        color_list = [(1.0000, 0.6636, 0.3583),  # 4000K
             (1.0000, 0.7992, 0.6045),  # 5000K
             (1.0000, 0.9019, 0.8473),  # 6000K
             (0.9337, 0.9150, 1.0000),  # 7000K
             (0.7874, 0.8187, 1.0000),  # 8000K
             (0.6693, 0.7541, 1.0000),  # 9000K
             (1.0,1.0,1.0) # white
             ]
        idx = random.randint(0, len(color_list)-1)
        return color_list[idx]
    
