from pathlib import Path
import numpy as np
import pandas as pd
f_lst = Path("./").rglob("pull_1_pullf.xvg")
lst = []
arr = np.array([])
for fname in f_lst:
    try:
        d = np.loadtxt(fname,comments=["#","@"])[:,1].max()*1.6
        lst.append([fname.parent.name,d])
        arr = np.append(arr,d)
    except:
        continue
print(arr.max(),arr.mean(),arr.min())
df = pd.DataFrame(lst,columns=["name","max_force"])
df.to_csv("./step2_smd_result.csv")