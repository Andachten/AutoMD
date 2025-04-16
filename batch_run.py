from multiprocessing import Pool
from pathlib import Path
import shutil
from md_utils_dev import MD
import os
from os import system
import time
import hydra



CWD = os.getcwd()

def run(fname, config, i):
    #创建工作文件夹，并强制转移至主文件夹
    os.chdir(CWD)
    system(f"mkdir -p {fname.stem}")
    
    #准备力场文件以及移动结构文件
    if not Path(f"./{fname.stem}/charmm36-jul2022.ff").is_dir():
        shutil.copytree("./charmm36-jul2022.ff", f"./{fname.stem}/charmm36-jul2022.ff")
    if not Path(f"./{fname.stem}/{fname.name}").is_file():
        shutil.copy(f"{str(fname)}", f"./{fname.stem}/{fname.name}")
    
    #更换工作路径至工作文件夹
    os.chdir(f"{fname.stem}")
    
    #设定初始gpu编号，并检测gpu使用情况
    gpu_lst = config.global_settings.gpu_lst
    gpu_id = gpu_lst[i % len(gpu_lst)]
    gpu_flag = Path(f"../{gpu_id}gpu")
    count = -1
    while gpu_flag.is_file():
        count += 1
        time.sleep(5)
        gpu_id = gpu_lst[count%len(gpu_lst)]
        gpu_flag = Path(f"../{gpu_id}gpu")
        if count > 16:
            count = -1
    #创建gpu使用信号
    with open(gpu_flag, 'w') as f:
        f.write('')
    
    try:
        MD(config, Path(fname.name), gpu_id)
    except Exception as err:
        print(err)
    
    #撤销gpu使用信号
    system(f"rm {str(gpu_flag)}")
    os.chdir("..")

@hydra.main(version_base=None, config_path="", config_name="config")  
def main(config):
    gpu_lst = config.global_settings.gpu_lst
    f_lst = list(Path("./").glob("*.pdb"))
    process = []
    pool = Pool(len(gpu_lst))
    for i, fname in enumerate(f_lst):
        process.append(pool.apply_async(run, (fname, config, i, )))
    pool.close()
    pool.join()
    for res in process:
        result = res.get()

@hydra.main(version_base=None, config_path="", config_name="config")  
def main_test(config):
    print(config.global_settings.gpu_lst)

if __name__ == '__main__':
    main()
    #main_test()
