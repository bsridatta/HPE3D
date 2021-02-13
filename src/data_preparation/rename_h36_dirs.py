import glob
import os

h5_dir = f"{os.getenv('HOME')}/lab/HPE_datasets/h36m_h5s/"

for root, dirs, files in os.walk(h5_dir):
    if "StackedHourglass" in root:
        prefix_wa = sorted(list(set([x.split(".")[0] for x in files])))
        prefix = sorted(list(set([x.split(".")[0].split("_")[0] for x in files])))

        map = {}
        for p in prefix:
            i = 1
            for pa in prefix_wa:
                if p in pa:
                    map[pa] = p+"_0"+str(i)
                    i += 1

        for file in files:
            new = file.split(".")
            new[0] = map[new[0]]
            os.rename(root+"/"+file, root+"/"+ ".".join(new))
