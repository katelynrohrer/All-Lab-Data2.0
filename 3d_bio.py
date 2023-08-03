import os
import json
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from icecream import ic
from progressbar import progressbar 


def prompt(command: str) -> str:
    """Returns output of given terminal command as a string."""
    with os.popen(command) as result:
        return result.read().strip()


def get_points(filename: str) -> np.ndarray[float]:
    """
    Given a csv filename, aggregates all 3d points in the file into a numpy
    array of the shape (_, 3)
    """
    df = pd.read_csv(filename)

    # Find x,y,z columns in file
    columns = list(df.columns)
    x_axes = [col for col in columns if " x" in col.lower()]
    y_axes = [col for col in columns if " y" in col.lower()]
    z_axes = [col for col in columns if " z" in col.lower()]

    # Add all data points into one array
    points = []
    for x_data, y_data, z_data in zip(x_axes, y_axes, z_axes):
        x = df[x_data]
        y = df[y_data]
        z = df[z_data]
        points += list(zip(x, y, z))

    return np.array(points)


def get_volume(points: np.ndarray[float]) -> float:
    """
    Given a numpy array of the shape (_, 3) returns the volume of the polygon
    that encloses those points
    """
    hull = ConvexHull(points)
    vertices = points[hull.vertices, :]
    centroid = np.mean(vertices, axis=0)

    # Calculate volume, one tetrahedron at a time
    volume = 0
    for simplex in hull.simplices:
        # Each tetrahedron is a triangle from the 3d polygon plus the centroid
        simplex_vertices = points[simplex, :]
        simplex_vertices = np.vstack((simplex_vertices, centroid))
        tetrahedron_volume = (
            np.abs(
                np.dot(
                    simplex_vertices[0] - simplex_vertices[3],
                    np.cross(
                        simplex_vertices[1] - simplex_vertices[3],
                        simplex_vertices[2] - simplex_vertices[3],
                    ),
                )
            )
            / 6
        )
        volume += tetrahedron_volume
    return volume


def get_file_info(filename: str) -> Dict[str, any]:
    """Returns the info gathered from filename as a dict"""
    # Remove any directory
    filename = filename[filename.rindex('/') + 1:]

    labels = ["source", "motion", "subject", "muscle", "run"]
    info = {label: elem.lower() for label, elem in zip(labels, filename.split('.'))}

    info["speed"] = 'fast' if 'Fast' in info["run"] else 'slow'
    info["run"]   = int(info["run"][1])
    return info



def motion_envelope_files() -> List[str]:
    """
    Returns the list of linear displacement biostamp files whose muscle is the 
    designated one for that motion
    """
    motion_stamps = {
            "chestaa": "forearm",
             "shoulderfe": "forearm",
             "shoulderaa": "forearm",
             "bicepc": "forearm",
             "fingerp": "index",
             "bodyl": "cspine"
         }

    file_list = get_csvs_in_dir("./Data")
    file_list = (file.strip('./').split('.') for file in file_list)

    return [
        '.'.join(file)
        for file in file_list
        if has_pair(motion_stamps, file[1], file[3]) and 'Linear' in file
    ]


def has_pair(dict: Dict[str, str], key: str, val: str) -> bool:
    """ True if dict has given key/value pair, False otherwise. Case insensitive."""
    return key.lower() in dict and dict[key.lower()] == val.lower()


def graph_polygon(points: np.ndarray[float], ax, **kwargs) -> None:
    """
    Given a numpy array of the shape (_, 3), graphs a 3d polygon which
    wraps those 3d points.
    """
    hull = ConvexHull(points)

    # Each simplex is an array of 3 indeces for points that make up one of
    # the triangles of the polygon
    for simplex in hull.simplices:
        simplex_vertices = points[simplex, :]
        ax.plot_trisurf(
            simplex_vertices[:, 0],
            simplex_vertices[:, 1],
            simplex_vertices[:, 2],
            linewidth=0.2,
            edgecolor="black",
            alpha=0.5,
            **kwargs
        )

def get_csvs_in_dir(dir: str) -> List[str]:
    """Returns a list of all csv filenames in a given directory"""
    retlist = []
    for root, _, files in os.walk(dir):
        retlist += [os.path.join(root, file) for file in files if file.endswith(".csv")]
    return retlist

def search(*terms: str, dir='./Data') -> List[str]:
    """Returns a list of files in dir which contain all given terms (case insensitive)."""
    files = get_csvs_in_dir(dir)
    # A list of all words from given search terms
    terms = [word for string in terms for word in string.split()]
    return [
            file
            for file in files
            if all(term.lower() in file.lower() for term in terms)
        ]


def get_config() -> Dict[str, str]:
    """
    Attempts to load a settings.json file. If none exists, creates one and 
    prompts user to enter preferences to store
    """
    config_file = 'settings.json'

    required_fields = ["data_folder"]
    config = {}
    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            config = json.load(file)

    for field in required_fields:
        if field not in config:
            config[field] = input(f"Enter value for {field}: ")

    with open(config_file, 'w') as f:
        json.dump(config, f)

    return config


def main():
    config = get_config()
    os.chdir(config["data_folder"])

    # Uncomment to disable debug printing
    # ic.disable()
    ic.configureOutput(includeContext=True)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # file1 = search("biostamp cg1f chestaa r1fast linear bicep")[0]
    # file2 = search("biostamp cg1f chestaa r2fast linear bicep")[0]
    # points1 = get_points(file1)
    # points2 = get_points(file2)
    # volume1 = get_volume(points1)
    # volume2 = get_volume(points2)
    # print("RUN 1 VOLUME: ", volume1)
    # print("RUN 2 VOLUME: ", volume2)
    # print("PERCENT CHANGE VOLUME: ", (volume2 - volume1)/volume2)
    # graph_polygon(points1, ax, color='blue')
    # graph_polygon(points2, ax, color='red')
    # plt.show()

    df = pd.DataFrame()
    files = motion_envelope_files()
    for file in progressbar(files):
        points = get_points(file)
        volume = get_volume(points)
        info   = get_file_info(file)
        info["volume"] = volume
        info = {key: [val] for key, val in info.items()}
        file_data = pd.DataFrame(info)
        df = pd.concat([df, file_data])
    df.to_csv("motion_envelope_area.csv")

    # df = pd.read_csv("motion_envelope_area.csv")

    # plt.grid(True)
    # grouped_df = df.groupby("motion")["volume"].mean()
    # ic(grouped_df)
    # plt.bar(grouped_df.index, grouped_df.values)
    # plt.title("Average volume per motion")
    # plt.show()

    # plt.grid(True)
    # grouped_df = df.groupby(["subject", "motion"])["volume"].var()
    # ic(grouped_df)
    # avg_motion_variance = grouped_df.groupby("motion").mean()
    # plt.bar(avg_motion_variance.index, avg_motion_variance.values)
    # plt.title("Average variance within each subject for each motion")
    # plt.show()

    # plt.grid(True)
    # grouped_df = df.groupby(["subject", "motion"])["volume"].std()
    # ic(grouped_df)
    # avg_motion_variance = grouped_df.groupby("motion").mean()
    # plt.bar(avg_motion_variance.index, avg_motion_variance.values)
    # plt.title("Average standard deviation within each subject for each motion")
    # plt.show()

if __name__ == "__main__":
    main()
