# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
from os import system
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBIO
from Bio.PDB.vectors import Vector
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter
from pathlib import Path
import numpy as np
import copy
import hydra

version = '0.0.1'

PULL_MDP = {'define': '-DPOSRES_PULL',
            'integrator ': ' md',
            'dt': '0.002',
            'nsteps': '50000000',
            'nstxout': '50000',
            'nstvout': '50000',
            'nstfout': '50000',
            'nstcalcenergy': '5000',
            'nstenergy': '5000',
            'nstlog': '5000',
            'cutoff-scheme': 'Verlet',
            'nstlist': '40',
            'rlist': '1.2',
            'vdwtype': 'Cut-off',
            'vdw-modifier': 'Force-switch',
            'rvdw_switch': '1.0',
            'rvdw': ' 1.2',
            'coulombtype': 'PME',
            'rcoulomb': '1.2',
            'tcoupl': ' V-rescale',
            'tc_grps': 'SOLU SOLV',
            'tau_t': '1.0 1.0',
            'ref_t': '298 298',
            'constraints': 'h-bonds',
            'constraint_algorithm': 'LINCS',
            'continuation': ' yes',
            'nstcomm': '100',
            'comm_mode': ' None',
            'comm_grps': 'system',
            'pull': 'yes',
            'pull-ngroups': '1',
            'pull-ncoords ': '1',
            'pull-group1-name': 'PULL',
            'pull-coord1-type': 'umbrella',
            'pull-coord1-geometry': 'direction-periodic',
            'pull-coord1-groups': '0 1',
            'pull-coord1-vec': '0.0000 0.0000 1.0000',
            'pull-coord1-rate': '0.001',
            'pull-coord1-init': '0',
            'pull-coord1-start': 'yes',
            'pull_coord1_k': '1000',
            'pull_coord1_dim': 'N N Y'}
MINI_MDP = {
    'define':'-DPOSRES',
    'integrator' : 'steep',
    'emtol':'700.0',
    'nsteps':'5000',
    'nstlist':'10',
    'cutoff-scheme':'Verlet',
    'rlist':'1.2',
    'vdwtype':'Cut-off',
    'vdw-modifier':'Force-switch',
    'rvdw_switch':'1.0',
    'rvdw':'1.2',
    'coulombtype':'PME',
    'rcoulomb':'1.2',
    'constraints':'h-bonds',
    'constraint_algorithm':'LINCS'
    }
EQUI_MDP = {'define': '-DPOSRES',
            'integrator': 'md',
            'dt': '0.001',
            'nsteps': '125000',
            'nstxtcout': '5000',
            'nstvout': '5000',
            'nstfout': '5000',
            'nstcalcenergy': '100',
            'nstenergy': '1000',
            'nstlog': '1000',
            'cutoff-scheme': 'Verlet',
            'nstlist': '40',
            'rlist': '1.2',
            'vdwtype': 'Cut-off',
            'vdw-modifier': 'Force-switch',
            'rvdw_switch': '1.0',
            'rvdw': '1.2',
            'coulombtype': 'PME',
            'rcoulomb': '1.2',
            'tcoupl': 'V-rescale',
            'tc_grps': 'SOLU SOLV',
            'tau_t': '1.0 1.0',
            'ref_t': '298 298',
            'constraints': 'h-bonds',
            'constraint_algorithm': 'LINCS',
            'nstcomm': '100',
            'comm_mode': 'linear',
            'comm_grps': 'SOLU SOLV',
            'gen-vel': 'yes',
            'gen-temp': '298',
            'gen-seed': '-1'}
PROD_MDP = {'integrator': 'md',
            'dt': '0.002',
            'nsteps': '350000000',
            'nstxtcout': '100000',
            'nstvout': '100000',
            'nstfout': '100000',
            'nstcalcenergy': '5000',
            'nstenergy': '5000',
            'nstlog': '2000',
            'cutoff-scheme': 'Verlet',
            'nstlist': '40',
            'vdwtype': 'Cut-off',
            'vdw-modifier': 'Force-switch',
            'rvdw_switch': '1.0',
            'rvdw': '1.2',
            'rlist': '1.2',
            'rcoulomb': '1.2',
            'coulombtype': 'PME',
            'tcoupl': 'V-rescale',
            'tc_grps': 'SOLU SOLV',
            'tau_t': '1.0 1.0',
            'ref_t': '298 298',
            'pcoupl': 'C-rescale',
            'pcoupltype': 'isotropic',
            'tau_p': '5.0',
            'compressibility': '4.5e-5',
            'ref_p': '1.0',
            'constraints': 'h-bonds',
            'constraint_algorithm': 'LINCS',
            'continuation': 'yes',
            'nstcomm': '100',
            'comm_mode': 'linear',
            'comm_grps': 'SOLU SOLV'}
Annealing_mdp = {
        "annealing" :"single single",
        "annealing-npoints":"3 3",
        "annealing-time":"0 50000 1000000 0 50000 1000000",
        "annealing-temp":"0 310 410 0 310 410"}

def orientation_z(model, v1):
    v2 = Vector([0, 0, 1])
    vector1 = v1 / np.linalg.norm(v1)
    vector2 = v2 / np.linalg.norm(v2)
    vector1 = np.array(list(vector1)).reshape([1,3])
    vector2 = np.array(list(vector2)).reshape([1,3])
    print(vector1,vector2)
    rotation_matrix = Rotation.align_vectors(vector1, vector2)[0].as_matrix()
    model.transform(rotation_matrix,[0,0,0])
    return model

def write_structure(structure,fname):
    io = PDBIO()
    io.set_structure(structure)
    io.save(fname)

def get_structuresize(model):
    all_atom_lst = []
    for chain in model.get_list():
        for res in chain.get_list():
            all_atom_lst += [atom.get_coord() for atom in res]
    all_atom_arr = np.vstack(all_atom_lst)
    max_ = all_atom_arr.max(axis=0)
    min_ = all_atom_arr.min(axis=0)
    size = (max_-min_)*0.1
    return size

def parser_index(fname,ndx=''):
    if ndx=='':
        with os.popen("echo 'q\n'|gmx make_ndx -f {} -o index.ndx".format(fname)) as f:
            x = f.readlines()
    else:
        with os.popen("echo 'q\n'|gmx make_ndx -f {} -o index.ndx -n {}".format(fname,ndx)) as f:
            x = f.readlines()
    start = False
    lst = []
    for s in x:
        if 'System' in s:
            start = True
        if 'nr' in s:
            break
        if start and s=='\n':
            break
        if start:
            index,name = int(s.split()[0]),s.split()[1]
            lst.append([index,name])
    return lst

def write_mdp(dic,fname):
    s = ''
    for k,v in dic.items():
        s += "{} = {}\n".format(k,v)
    with open(fname,'w') as f:
        f.write(s)
def write_log(fname,message,overwrite=False):
    if overwrite:
        s = 'w'
    else:
        s = 'a'
    with open(fname,s) as f:
        f.write(message)

def parse_split_group(s):
    s = s.upper()
    groups_lst = [[],[]]
    for i,g in enumerate(s.split("|")):
        for x in g.split(","):
            chain_name,resid,atom_name = x.split(":")
            groups_lst[i].append([chain_name,resid,atom_name])
    return groups_lst

def distance_3d_np(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)

def smd_pull_fix(model,groups_lst):
    chain_name_lst = []
    chain_lst  = []
    for chain in model.get_list():
        chain_name_lst.append(chain.get_id())
        chain_lst.append(chain)
    pull_fix_group = [[],[]]
    for i,g in enumerate(groups_lst):
        for x in g:
            chain_index = chain_name_lst.index(x[0])
            chain = chain_lst[chain_index]
            for_len = sum(len(chain.get_list()) for chain in chain_lst[:chain_index])
            resid = int(x[1])
            if resid>0:
                result = for_len + resid
            else:
                result = for_len + resid +len(chain.get_list()) + 1
            pull_fix_group[i].append(result)
    return pull_fix_group

def write_fix_top():
    with open(r'./topol.top', 'r') as f:
        text = f.read()
    if 'FIX.itp' in text:
        return None
    with open(r'./topol.top', 'r') as f:
            text = f.readlines()
    lst = []
    for i, x in enumerate(text):
        if '; Include water topology' in x:
            lst.append("#ifdef POSRES_PULL\n")
            lst.append('#include "FIX.itp"\n')
            lst.append("#endif\n\n")
        lst.append(x)
    with open('./topol.top', 'w') as f:
        f.write(''.join(lst))

def get_gmx_version():
    res = os.popen("gmx --version")
    text_lst = res.read().split("\n")
    for text in text_lst:
        if 'GROMACS version' in text:
            version = float(text.split(":")[1].replace(" ",""))
            break
    return version

def rmsd_calc(name,time_step,config):
    if Path(f"{name}.tpr").is_file() and Path(f"{name}.xtc").is_file():
        system(f"echo '1\n1\n'|gmx trjconv -s {name}.tpr -f {name}.xtc -o {name}_noPBC.xtc -n index.ndx -pbc cluster")
        system(f"echo '4\n4\n'|gmx rms -s {name}.tpr -f {name}_noPBC.xtc -n index.ndx -o rmsd_temp.xvg")
    else:
        return None
    rmsd_mean,rmsd_max,rmsd_std = 0,0,0
    if Path('rmsd_temp.xvg').is_file():
        rmsd = np.loadtxt('rmsd_temp.xvg',comments=["#","@"])
        rmsd_smooth = gaussian_filter(rmsd[:,1],3)
        rmsd_mean = rmsd_smooth.mean()
        rmsd_max = rmsd_smooth.max()
        rmsd_std = rmsd_smooth.std()
        with open(f'{name}_rmsd_calc_at_{time_step}.txt','w') as f:
            f.write(f"Mean:{rmsd_mean}; Max: {rmsd_max}; std: {rmsd_std}\n")
    early_stop = False
    j = 0
    if config.MD_settings.early_stop:
        if config.MD_settings.early_stop_type == 'rmsd_max':
            j = rmsd_max
        if config.MD_settings.early_stop_type == 'rmsd_mean':
            j = rmsd_mean
        if config.MD_settings.early_stop_type == 'rmsd_std':
            j = rmsd_std
        if j > config.MD_settings.early_stop_threshold:
             early_stop = True
    if early_stop:
        with open(f'{name}_earlystop','w') as f:
            f.write(f"Time step: {time_step}; Mean:{rmsd_mean}; Max: {rmsd_max}; std: {rmsd_std}\n")
def get_current_step(name):
    system(f"gmx convert-trj -f {name}.cpt -s {name}.tpr -o temp.gro")
    with open('temp.gro','r') as f:
        x = f.readlines()
    time = int(x[0].replace("\n","").replace(" ","").split("=")[1])/1000
    #system("rm temp.gro")
    return time

def get_annealing_mdp_dic(Annealing_process):
    annealing_mdp = copy.deepcopy(Annealing_mdp)
    points = len(Annealing_process[0])
    time_ = [str(t*1000) for t in Annealing_process[0]]
    time_ = ' '.join(time_)
    temp_ = [str(t) for t in Annealing_process[1]]
    temp_ = ' '.join(temp_)
    annealing_mdp['annealing-npoints'] = f"{points} {points}"
    annealing_mdp['annealing-time'] = f"{time_} {time_}"
    annealing_mdp['annealing-temp'] = f"{temp_} {temp_}"
    return annealing_mdp

def check_file(config):
    if not config.global_settings.over_write:
        return False
    if config.SMD_settings.run:
        if Path("./pull_1.tpr").is_file():
            return True
    elif config.MD_settings.run:
        if Path("./md_1.tpr").is_file():
            return True
    if config.MD_settings.run and config.MD_settings.skip_smd:
        if Path("./md_1.tpr").is_file():
            return True
    return False


def step1_preparation(config, fname='./protein.pdb'):
    if check_file(config):
        return None

    #准备能量最小化文件
    mini_mdp = copy.deepcopy(MINI_MDP)
    write_mdp(mini_mdp, r'./mini.mdp')

    #初始结构读取
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', fname)
    #只读第一个model
    model = structure[0]

    #根据SMD设置进行分子方向的重整
    #对SMD的拉伸组和固定组进行解析，groups[0]中包含拉伸组所有原子，groups[1]包含固定组所有原子
    if config.SMD_settings.run:
        groups = [[],[]]
        groups_lst = parse_split_group(config.SMD_settings.group_split)
        for i,g in enumerate(groups_lst):
            for x in g:
                chain_name,resid,atom_name = x
                if resid != '':
                    resid  = int(resid)
                chain = [chain for chain in model.get_chains() if chain.get_id()==chain_name][0]
                res_lst = chain.get_list()
                if resid != '':
                    resid  = res_lst[resid].get_id()[1]
                if resid == '':
                    groups[i] += sum([res.get_list() for res in chain],[])
                    continue
                res = [res for res in res_lst if res.get_id()[1]==resid][0]
                if atom_name == '':
                    groups[i] += res.get_list()
                    continue
                groups[i].append(res[atom_name])
        #计算两个组之间的向量
        coord_lst = [np.vstack([atom.get_coord() for atom in g]) for g in groups]
        coord_mean_lst = [coord.mean(axis=0) for coord in coord_lst]
        v1 = Vector(coord_mean_lst[0]-coord_mean_lst[1])

        #旋转结构以便拉伸,第二个参数是方向向量
        model = orientation_z(model,v1)

        #计算两组之间的距离（折叠状态）
        distance = distance_3d_np(coord_mean_lst[0],coord_mean_lst[1])*0.1

        #计算ΔLc，仅在蛋白解折叠适用，且只针对第一条链
        if config.SMD_settings.type == 'unfolding':
            aa_num = len(model.get_list()[0].get_list())
            total_length = aa_num * 0.365 - distance
            if config.SMD_settings.pull_distance > 0:
                pulling_length = config.SMD_settings.pull_distance
            else:
                pulling_length = total_length * config.SMD_settings.pull_rotio
        elif config.SMD_settings.type == 'unbinding':
            pulling_length = config.SMD_settings.pull_distance
    else:
        pulling_length = 0

    write_structure(model, './input.pdb')

    #获取体系大小
    size = get_structuresize(model)
    #设置格子大小
    box_pad = config.global_settings.box_pad
    # 正反两个方向
    box_pad = box_pad * 2 
    if config.SMD_settings.run:
        x_box, y_box, z_box = size + box_pad
        z_box = z_box + pulling_length
    else:
        x_box, y_box, z_box = [size.max()+box_pad]*3

    #解决charmm36力场对于NH2-MET的报错
    first_resname_lst = [chain.get_list()[0].get_resname() for chain in model.get_list()]
    ter_cmd = "1\n1\n"
    for resname in first_resname_lst:
        if resname == 'MET':
            ter_cmd += '1\n0\n'
        else:
            ter_cmd += '0\n0\n'
    
    #生成拓扑文件
    system(f"echo '{ter_cmd}'| gmx pdb2gmx -f input.pdb -ignh -o input.gro -ter")

    #生成盒子，溶剂化
    if config.SMD_settings.run:
        system(f"gmx editconf -f input.gro -o input.gro -box {x_box} {y_box} {z_box} -center {x_box*0.5} {y_box*0.5} {size[2]*0.5+0.5}")
    else:
        system(f"gmx editconf -f input.gro -o input.gro -box {x_box} {y_box} {z_box} -center {x_box*0.5} {y_box*0.5} {z_box*0.5}")
    system("gmx solvate -cp input.gro -cs spc216.gro -o input.gro -p topol.top ")
    system("gmx grompp -f mini.mdp -c input.gro -r input.gro -p topol.top -o ions.tpr -maxwarn 1")
    system("echo 'SOL\n'|gmx genion -s ions.tpr -o input.gro -p topol.top -pname SOD -nname CLA -neutral -conc 0.15")

    #索引操作
    index_lst = parser_index("input.gro")
    last_index = index_lst[-1][0]

    #索引操作命令行积累
    #设定溶剂组和溶质组
    index_cmd = ''
    index_cmd += f"1\n"
    index_cmd += f"name {last_index + 1} SOLU\n"
    last_index = last_index + 1

    index_cmd += f"! {last_index}\n"
    index_cmd += f"name {last_index + 1} SOLV\n"
    last_index = last_index + 1

    #定义拉伸组和固定组，由于gmx make_ndx的限制，目前只能固定蛋白的CA碳
    if config.SMD_settings.run:
        pull_fix_group = smd_pull_fix(model, groups_lst)
        pull_group = [str(i) for i in pull_fix_group[0]]
        fix_group = [str(i) for i in pull_fix_group[1]]
        pull_group_resid = ' '.join(pull_group)
        fix_group_resid = ' '.join(fix_group)

        index_cmd += f"ri {fix_group_resid} & a ca\n"
        index_cmd += f"name {last_index+1} FIX\n"
        last_index = last_index + 1
        index_cmd += f"ri {pull_group_resid} & a ca\n"
        index_cmd += f"name {last_index+1} PULL\n"

    index_cmd += 'q\n'
    last_index = last_index + 1
    #生成索引
    system(f"echo '{index_cmd}'|gmx make_ndx -f input.gro -o index.ndx")

    #生成平衡模拟mdp文件
    equi_mdp = copy.deepcopy(EQUI_MDP)
    temp = config.MD_settings.temp
    equi_mdp['ref_t'] = f"{temp} {temp}"
    write_mdp(equi_mdp, r'./equi.mdp')


    #生成拉伸mdp文件
    if config.SMD_settings.run:
        system("echo 'FIX'|gmx genrestr -f input.gro -o FIX.itp -n index.ndx")
        write_fix_top()
        pull_speed = config.SMD_settings.speed
        pull_time = pulling_length / pull_speed
        if config.SMD_settings.time>0:
            pull_time = config.SMD_settings.time
        temp = config.MD_settings.temp
        pull_mdp = copy.deepcopy(PULL_MDP)
        pull_mdp['ref_t'] = f"{temp} {temp}"
        pull_mdp['nsteps'] = str(int(pull_time/2e-6))
        pull_mdp['pull-coord1-rate'] = str(pull_speed/1e3)

        write_mdp(pull_mdp, './pull.mdp')
    
    #生成成品模拟mdp文件
    prod_mdp = copy.deepcopy(PROD_MDP)
    temp = config.MD_settings.temp
    time = config.MD_settings.time
    prod_mdp['ref_t'] = f"{temp} {temp}"
    prod_mdp['nsteps'] = str(int(time / 2e-6))
    if config.MD_settings.Annealing:
        Annealing_process = config.MD_settings.Annealing_process
        annealing_mdp = get_annealing_mdp_dic(Annealing_process)
        prod_mdp.update(annealing_mdp)
        prod_mdp["pcoupl"] = 'Berendsen'
    if config.MD_settings.Fix:
        prod_mdp.update({'define': '-DPOSRES_PULL'})
        prod_mdp.update({"refcoord_scaling":'com'})

    write_mdp(prod_mdp, r'./md.mdp')

    system("rm *#*")

def mini_run(config, gpu_id=0):
    if Path('./mini.gro').is_file() and not config.global_settings.over_write:
        return None
    system("gmx grompp -f mini.mdp -o mini.tpr -c input.gro -r input.gro -p topol.top -n index.ndx")
    system(f"gmx mdrun -v -deffnm mini  -ntmpi 1 -ntomp 8 -pin on -pinoffset {gpu_id*16} -gpu_id {gpu_id}")

def equi_run(config, gpu_id=0):
    if Path("./equi.gro").is_file() and not config.global_settings.over_write:
        return None
    bonded = config.global_settings.gmx_bonded
    update = config.global_settings.gmx_update
    system("gmx grompp -f equi.mdp -o equi.tpr -c mini.gro -r mini.gro -p topol.top -n index.ndx")
    system(f"gmx mdrun -v -deffnm equi  -ntmpi 1 -ntomp 8 -gpu_id {gpu_id} -pin on -pinoffset {gpu_id*16} -update gpu  -notunepme -bonded {bonded}")

def md_run(config, gpu_id=0):
    if not config.MD_settings.run:
        return None
    times = config.MD_settings.times
    bonded = config.global_settings.gmx_bonded
    update = config.global_settings.gmx_update
    if config.global_settings.over_write:
        system("rm md*.cpt")
        system("rm md*.tpr")
        system("rm *.done")
        system("rm md*rmsd_calc*.txt")
    for i in range(times):
        i = i+1
        name = f"md_{i}"
        if not Path(f"{name}.cpt").is_file():
            system(f"gmx grompp -f md.mdp -o {name}.tpr -c equi.gro -r equi.gro -p topol.top -n index.ndx -maxwarn 1")
        run_scheduler = [config.MD_settings.time]
        if config.MD_settings.early_stop:
            md_time = config.MD_settings.time
            early_stop_step = config.MD_settings.early_stop_step
            start_step = early_stop_step
            run_scheduler = list(range(start_step,md_time,early_stop_step))+run_scheduler
        for time_step in run_scheduler:
            #早停终止判断
            if config.MD_settings.early_stop:
                start_time_step = time_step-early_stop_step
                if not Path(f"{name}_rmsd_calc_at_{start_time_step}.txt").is_file():
                    rmsd_calc(name, time_step-early_stop_step, config)
                if Path(f"{name}_earlystop").is_file():
                    break
            system(f"gmx convert-tpr -s {name}.tpr -o {name}.tpr -until {time_step*1000}")
            system("rm *#*")
            if Path(f'{name}.cpt').is_file():
                system(f"gmx mdrun -v -deffnm {name}  -ntmpi 1 -ntomp 8 -gpu_id {gpu_id} -pin on -pinoffset {gpu_id * 16} -update {update}  -notunepme -bonded {bonded} -cpi {name}.cpt")
            else:
                system(f"gmx mdrun -v -deffnm {name}  -ntmpi 1 -ntomp 8 -gpu_id {gpu_id} -pin on -pinoffset {gpu_id * 16} -update {update}  -notunepme -bonded {bonded}")
        if config.MD_settings.early_stop:
            rmsd_calc(name, start_time_step, config)
            
def smd_run(config, gpu_id=0):
    if not config.SMD_settings.run or config.MD_settings.skip_smd:
        return None
    times = config.SMD_settings.times
    bonded = config.global_settings.gmx_bonded
    update = config.global_settings.gmx_update
    if config.global_settings.over_write:
        system("rm pull*.cpt")
        system("rm pull*.tpr")
    for i in range(times):
        name = f'pull_{i+1}'
        if Path(f"{name}.cpt").is_file():
            system(f"gmx mdrun -v -deffnm {name}  -ntmpi 1 -ntomp 8 -gpu_id {gpu_id} -pin on -pinoffset {gpu_id * 16} -update {update} -notunepme -bonded {bonded} -cpi {name}.cpt -px {name}_pullx.xvg -pf {name}_pullf.xvg")
        else:
            system(f"gmx grompp -f pull.mdp -o {name}.tpr -c md_1.gro -r md_1.gro -p topol.top -n index.ndx")
            system(f"gmx mdrun -v -deffnm {name}  -ntmpi 1 -ntomp 8 -gpu_id {gpu_id} -pin on -pinoffset {gpu_id*16} -update {update}  -notunepme -bonded {bonded}")

def MD(config, fname, gpu_id=0):
    system("rm *#*")
    step1_preparation(config, fname)
    mini_run(config, gpu_id)
    equi_run(config, gpu_id)
    md_run(config, gpu_id)
    smd_run(config, gpu_id)
    system("rm *#*")

@hydra.main(version_base=None, config_path="", config_name="config")
def main(config):
    fname = './protein.pdb'
    MD(config, fname, gpu_id=5)

if __name__ == '__main__':
    main()
