import os
from glob import glob
from progressbar import progressbar


def main():
    origin = "/Users/jordan/Documents/Work/All-Lab-Data/Lab/"
    newdir = "~/Documents/Work/All-Lab-Data2.0/Data/"

    files = glob(origin + "**", recursive=True)
    files = list(filter(lambda x: "MOCA" in x and "csv" in x, files))
    changed_files = map(convert_file,files)
    changed_files = [filename[len(origin):] for filename in changed_files]
    for i in progressbar(range(len(files)), redirect_stdout=True):
        new_path = newdir + cleanse_path(changed_files[i])
        # print(f"going to move {files[i]} to {new_path}")
        os.popen(f"cp {files[i]} {new_path}")



def convert_file(filename):
    path_split = filename.split(os.sep)
    trial_info = path_split[-1].split(".")
    trial_info = [part for part in trial_info if part != "" and part != "csv"]
    try:
        motion, camera, subject, run, _, _, speed, _ = trial_info
    except:
        print(f"failed to convert {filename}")
    run = f"R{run[-1]}{speed}"
    if motion == "BodyLean":
        motion = "BodyL"
    small_name = f"MOCA.{motion}.{subject}.{camera}.{run}.All.csv"
    rest = os.sep.join(path_split[:-1])
    return rest + os.sep + small_name

def cleanse_path(filename):
    dict = {
            "ShoulderAbduction": "ShoulderAA",
            "BodyLean": "BodyL",
            "ShoulderFlexion": "ShoulderFE",
            "ShoulderFlexion": "ShoulderFE",
            "BicepCurl": "BicepC",
            "FingerPinch": "FingerP",
            "ChestAA": "ChestAA",
    }
    path_split = filename.split(os.sep)
    experiment = path_split[0]
    experiment = experiment[experiment.index('_') + 1:]
    run = path_split[2]
    run = f"R{run[5]}{run[7:]}"
    path_split[0] = dict[experiment]
    path_split[2] = run
    return os.sep.join(path_split)

        

if __name__ == "__main__":
    main()

