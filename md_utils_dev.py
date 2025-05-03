# Create by ZBin
# 2024/09/11

import copy
import hydra
from pathlib import Path
import numpy as np
import MDAnalysis as mda
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R
from os import system
import subprocess
import time

PULL_MDP = {'define': '-DFIX',
            'integrator ': ' md',
            'dt': '0.002',
            'nsteps': '50000000',
            'nstxout': '50000',
            'nstxtcout':'50000',
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


def get_gpu_info():
    # 调用 nvidia-smi 命令
    command = "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
    result = subprocess.run(command, stdout=subprocess.PIPE, shell=True, text=True)
    
    # 解析 nvidia-smi 输出
    gpu_info = []
    for line in result.stdout.splitlines():
        index, name, mem_used, mem_total, util_gpu = line.split(', ')
        gpu_info.append({
            'index': int(index),
            'name': name,
            'mem_used': int(mem_used),
            'mem_total': int(mem_total),
            'util_gpu': int(util_gpu)
        })
    
    return gpu_info

def get_programs_on_gpu(gpu_index):
    # 调用 nvidia-smi 获取当前 GPU 上运行的程序
    command = f"nvidia-smi -i {gpu_index} --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits"
    result = subprocess.run(command, stdout=subprocess.PIPE, shell=True, text=True)
    
    programs = []
    for line in result.stdout.splitlines():
        pid, process_name, used_memory = line.split(', ')
        # 处理 used_memory 为 [N/A] 的情况
        if used_memory == '[N/A]':
            used_memory = 0  # 可以设置为 0 或者其他默认值
        else:
            used_memory = int(used_memory)
        
        programs.append({
            'pid': int(pid),
            'process_name': process_name,
            'used_memory': used_memory
        })
    return programs

def wait_for_gpu(gpu_id,sleep_time=20):
    wait_lst = ['gmx','esmfold','colab']
    while True:
        gpu_use_lst = get_programs_on_gpu(gpu_id)
        program = [x['process_name'] for x in gpu_use_lst]
        break_ = True
        for program_name in program:
            for x in wait_lst:
                if x in program_name:
                    break_ = False
        if break_:
            break
        time.sleep(sleep_time)

def write_mdp(dic,fname):
    s = ''
    for k,v in dic.items():
        s += "{} = {}\n".format(k,v)
    with open(fname,'w') as f:
        f.write(s)

def get_proteinsize(structure):
    protein = structure.select_atoms('protein')
    positions = protein.positions
    x_min, y_min, z_min = positions.min(axis=0)
    x_max, y_max, z_max = positions.max(axis=0)
    x_length = (x_max - x_min)/10
    y_length = (y_max - y_min)/10
    z_length = (z_max - z_min)/10
    return x_length, y_length, z_length

def parse_ndx(fname='index.ndx'):
    with open(fname,'r') as f:
        x = f.readlines()
    lst = [i.replace(' ','').replace('[','').replace(']','').replace('\n','') for i in x if "[" in i]
    lst = [[i,t] for i,t in enumerate(lst)]
    return lst

def append_ndx(ndx_append_lst,fname='index.ndx'):
    s = ""
    for name,index_lst in ndx_append_lst:
        s += f"[ {name} ]\n"
        lst = []
        for i,index in enumerate(index_lst):
            if i%15==0 and i!=0:
                lst.append('\n')
            lst.append(str(index))
        if lst[-1]!='\n':
            lst.append('\n')
        s += ' '.join(lst)
    with open(fname,'r') as f:
        x = f.readlines()
    text = f"{''.join(x[:-1])}\n{s}\n\n"
    with open(fname,'w') as f:
        f.write(text)
def parse_resid_chainID(selection, structure):
    """递归解析选择语法中的resid和chainID，并处理负数resid。"""
    if '(' in selection:
        # 处理括号嵌套
        stack = []
        new_sel = []
        for char in selection:
            if char == '(':
                stack.append(new_sel)
                new_sel = []
            elif char == ')':
                sub_sel = ''.join(new_sel)
                new_sel = stack.pop()
                new_sel.append(parse_resid_chainID(sub_sel, structure))
            else:
                new_sel.append(char)
        return ''.join(new_sel)
    else:
        # 处理不包含括号的选择
        sel_lst = selection.split()
        if 'resid' in sel_lst and 'chainID' in sel_lst:
            # 查找所有chainID和resid的位置
            chainID_indices = [i for i, s in enumerate(sel_lst) if s == 'chainID']
            resid_indices = [i for i, s in enumerate(sel_lst) if s == 'resid']

            for chainID_index, resid_index in zip(chainID_indices, resid_indices):
                chainID = sel_lst[chainID_index + 1]
                chain = next(chain for chain in structure.segments if chain.segid == chainID)
                resnum = len(chain.residues)
                resid_value = sel_lst[resid_index + 1]

                # 处理负数resid
                if resid_value.startswith('-'):
                    resid_value = resnum + int(resid_value) + 1
                    sel_lst[resid_index + 1] = str(resid_value)

            return ' '.join(sel_lst)
        else:
            return selection
        
def merge_top(fname='topol.top'):
    with open(fname,'r') as f:
        x = f.readlines()
    lst = []
    for text in x:
        if '#include' in text and 'forcefield' not in text:
            ipt_fname = eval([i for i in text.split(" ") if i != ''][-1])
            with open(ipt_fname,'r') as f:
                lst += f.readlines()
        else:
            lst.append(text)
    with open(fname,'w') as f:
        f.write(''.join(lst))

def parser_top(fname='topol.top'):
    import json
    with open(fname,'r') as f:
        x = f.readlines()
    x = [i.replace('\n','') for i in x if list(i)[0]!=';' and i!='\n' and '#' not in i]
    lst = []
    level1 = ['[ molecules ]','[ system ]','[ moleculetype ]']
    level2 = ['[ atoms ]','[ bonds ]','[ pairs ]','[ angles ]',
              '[ dihedrals ]','[ dihedrals ]','[ position_restraints ]',
              '[ cmap ]','[ dihedral_restraints ]','[ virtual_sitesn ]',
              '[ settles] ','[ exclusions ]','[ pairs_nb ]','[ distance_restraints ]',
              '[ constraints ]','[ virtual_sites1 ]','[ virtual_sites2 ]','[ virtual_sites3 ]',
              '[ virtual_sites4 ]', '[ distance_restraints ]','[ dihedral_restraints ]',
              '[ orientation_restraints ]','[ angle_restraints ]','[ angle_restraints_z ]']
    state = ''
    for i,text in enumerate(x):
        if text in level1:
            state = 'l1'
            lst.append([text,[]])
        elif text in level2:
            state = 'l2'
            lst[-1][-1].append([text,[]])
        else:
            if state == 'l1':
                lst[-1][-1].append(text)
            elif state == 'l2':
                lst[-1][-1][-1][-1].append(text)
    with open('topol.json','w') as f:
        json.dump(lst,f)
    return lst

def write_restraint_top(fix_dic,fc=1000,fname='topol.top'):
    with open(fname,'r') as f:
        x = f.readlines()
    
    mol_type = ''
    mol_type_read = False
    lst = []
    for text in x:
        if '[ moleculetype ]' in text:
            mol_type_read = True
            if mol_type in fix_dic.keys():
                lst.append("#ifdef FIX\n[ position_restraints ]\n")
                lst += [f"{i} 1 {fc} {fc} {fc}\n" for i in fix_dic[mol_type]]
                lst.append("#endif\n")
            lst.append(text)
            continue
        elif '[' in text:
            mol_type_read = False
        if list(text)[0] != ';' and '[' not in text and '#' not in text and mol_type_read and len(text.split())==2:
            mol_type = text.split()[0]
            mol_type_read = False
        lst.append(text)
    
    with open(fname,'w') as f:
        f.write(''.join(lst))

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

def early_stop_rmsd(name,time,config):
    fname = f"rmsd_{name}.xvg"
    reach_time = 0
    if Path(fname).is_file():
        rmsd_data = np.loadtxt(fname,comments=["#","@"])
        reach_time = rmsd_data[-1,0]/1e3+0.1 #ns
    if reach_time < time:
        if Path(f"{name}.tpr").is_file() and Path(f"{name}.xtc").is_file():
            system(f"echo '1\n1\n'|gmx trjconv -s {name}.tpr -f {name}.xtc -o {name}_noPBC.xtc -n index.ndx -pbc cluster")
            system(f"echo '4\n4\n'|gmx rms -s {name}.tpr -f {name}_noPBC.xtc -n index.ndx -o {fname}")
        else:
            return None
    rmsd_data = np.loadtxt(fname,comments=["#","@"])
    #保证时刻点正确
    rmsd = rmsd_data[:,1][np.where(rmsd_data[:,1]<=time*1e3)]
    rmsd_smooth = gaussian_filter(rmsd,3)
    rmsd_mean = rmsd_smooth.mean()
    rmsd_max = rmsd_smooth.max()
    rmsd_std = rmsd_smooth.std()
    if not Path(f'{name}_rmsd_calc_at_{time}.txt').is_file():
        with open(f'{name}_rmsd_calc_at_{time}.txt','w') as f:  
            f.write(f"Mean:{rmsd_mean}; Max: {rmsd_max}; std: {rmsd_std}\n")
    j = 0
    early_stop = False
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
            f.write(f"Time step: {time}; Mean:{rmsd_mean}; Max: {rmsd_max}; std: {rmsd_std}\n")




def step1_preparation(config, fname='./protein.pdb'):
    if Path("mini.tpr").is_file():
        return None
    
    gmx_exec = config.global_settings.gmx_exec

    mini_mdp = copy.deepcopy(MINI_MDP)
    write_mdp(mini_mdp, r'./mini.mdp')

    #结构读取
    structure = mda.Universe(fname)
    chain_num = len(structure.segments)

    #组选择语法负数解析
    group = config.MD_settings.group
    new_group = {}
    for name,sel in group.items():
        sel = parse_resid_chainID(sel, structure)
        new_group[name] = sel

    #SMD准备+方向调整
    if config.SMD_settings.run:
        fix_group = structure.select_atoms(new_group['FIX'])
        pull_group = structure.select_atoms(new_group['PULL'])
        fix_pos = fix_group.center_of_mass()
        pull_pos =  pull_group.center_of_mass()
        vector = pull_pos - fix_pos
        vector_normalized = vector / np.linalg.norm(vector)
        z_axis = np.array([0, 0, 1])
        rotation_vector = np.cross(vector_normalized, z_axis)
        rotation_angle = np.arccos(np.dot(vector_normalized, z_axis))
        if np.linalg.norm(rotation_vector) != 0:
            rotation_vector = rotation_vector / np.linalg.norm(rotation_vector)
            rotation = R.from_rotvec(rotation_angle * rotation_vector)
    
            # 对整个蛋白质施加旋转矩阵
            for atom in structure.atoms:
                atom.position = rotation.apply(atom.position)
        
        #拉伸组和固定组之间的距离
        distance = np.linalg.norm(fix_pos - pull_pos) / 10 #以nm计

        if config.SMD_settings.pull_distance > 0:
            pulling_length = config.SMD_settings.pull_distance
        else:
            resnum = len(list(structure.segments)[0].residues)
            total_length = resnum * 0.365 - distance
            pulling_length = total_length * config.SMD_settings.pull_rotio

    # 保存处理后的结构
    structure.atoms.write('input.pdb')
    protein_size = get_proteinsize(structure)

    #解决N端MET力场问题
    #获取每条链第一个残基名称
    first_resname_lst = [list(chain.residues)[0].resname for chain in structure.segments]
    last_resname_lst = [list(chain.residues)[-1].resname for chain in structure.segments]
    ter_cmd = "1\n1\n"
    for i,resname in enumerate(first_resname_lst):
        if first_resname_lst[i] == 'MET':
            ter_cmd += '1\n'
        else:
            ter_cmd += '0\n'
        if last_resname_lst[i] == 'MET':
            ter_cmd += '1\n'
        else:
            ter_cmd += '0\n'
    #生成拓扑文件
    system(f"echo '{ter_cmd}'| {gmx_exec} pdb2gmx -f input.pdb -ignh -o input.pdb -ter")

    #生成盒子
    box_pad = config.global_settings.box_pad
    box_type = config.global_settings.box_type
    if config.SMD_settings.run:
        x,y,z = protein_size
        x,y,z = x+box_pad,y+box_pad,z+box_pad
        z = z + pulling_length
        system(f"{gmx_exec} editconf -f input.pdb -o input.pdb -box {x} {y} {z} -center {x*0.5} {y*0.5} {protein_size[2]*0.5+0.5}")
    else:
        system(f"{gmx_exec} editconf -f input.pdb -o input.pdb -bt {box_type} -d {box_pad}")
    
    #溶剂化
    if config.global_settings.solvent == 'water':
        solvent = 'SOL'
        system(f"{gmx_exec} solvate -cp input.pdb -cs spc216.gro -o input.pdb -p topol.top ")
    elif config.global_settings.solvent == 'DMF':
        solvent = 'DMF'
        ...
    
    #离子化
    system(f"{gmx_exec} grompp -f mini.mdp -c input.pdb -r input.pdb -p topol.top -o ions.tpr -maxwarn 1")
    if config.global_settings.ion_conc > 0 :
        conc = config.global_settings.ion_conc
        system(f"echo '{solvent}\n'|{gmx_exec} genion -s ions.tpr -o input.pdb -p topol.top -pname SOD -nname CLA -neutral -conc {conc}")
    else:
        system(f"echo '{solvent}\n'|{gmx_exec} genion -s ions.tpr -o input.pdb -p topol.top -pname SOD -nname CLA -neutral")

    #索引文件操作, 将定义的组转换为ndx索引
    system(f"echo 'q\n'|{gmx_exec} make_ndx -f input.pdb -o index.ndx")
    chainIDs = list(set(list(structure.select_atoms('protein').chainIDs)))
    #离子化形成的structure
    input_structure = mda.Universe("input.pdb")
    if len(chainIDs)==1:
        chain = input_structure.select_atoms('protein')
        chain.chainIDs = [chainIDs[0]]*len(chain)
    #input_structure.atoms.write('input.pdb')

    ndx_append_lst = []
    for name,sel in new_group.items():
        sel_atom = input_structure.select_atoms(sel).ix + 1
        sel_atom = list(sel_atom)
        ndx_append_lst.append([name,sel_atom])
    append_ndx(ndx_append_lst,fname='index.ndx')

    #固定组操作
    #融合top文件，方便位置固定处理，2×
    merge_top()
    merge_top()

    #解析拓扑文件
    parsed_top = parser_top()
    molname_num = [i[1:] for i in parsed_top if i[0] == '[ molecules ]'][0][0]
    #mols = [['mol name', 'mol num'],...]
    molname_num = [[x for x in i.split(' ') if x != ''] for i in molname_num]
    atoms_num = [[(name,len(x[1][1][1:][0])) for x in parsed_top 
                if x[0]=='[ moleculetype ]' and name in x[1][0]]
                for name,n in molname_num]
    #从topol获取分子名×原子数，需要与结构文件中原子数对应
    top_index = []
    for i,v in enumerate(atoms_num):
        mol_name,atom_num = v[0]
        for mol_index in range(int(molname_num[i][1])):
            for k in range(atom_num):
                top_index.append([mol_name,k+1])
    assert len(input_structure.atoms) == len(top_index) , 'The atom number of top file is different grom structure file.'

    #mda索引从1开始
    fix_atoms = input_structure.select_atoms(new_group['FIX'])
    fix_atoms_index = fix_atoms.ix + 1
    #[(1877, ['Protein_chain_B', 157])
    fix_index_mol = [(i,top_index[i]) for i in fix_atoms_index-1]

    fix_dic = {}
    for v in fix_index_mol:
        mol_name = v[1][0]
        index = v[1][1]
        if mol_name not in fix_dic.keys():
            fix_dic[mol_name] = []
        if index not in fix_dic[mol_name]:
            fix_dic[mol_name].append(index)
    print(fix_dic)
    write_restraint_top(fix_dic,fc=1000,fname='topol.top')


    #模拟参数文件写入
    #NVT平衡模拟
    equi_nvt_mdp = copy.deepcopy(EQUI_MDP)
    temp = config.MD_settings.temp
    equi_nvt_mdp['ref_t'] = f"{temp} {temp}"
    write_mdp(equi_nvt_mdp, r'./equi_nvt.mdp')
    #NPT平衡模拟
    equi_npt_mdp = copy.deepcopy(PROD_MDP)
    temp = config.MD_settings.temp
    time = 1
    equi_npt_mdp['ref_t'] = f"{temp} {temp}"
    equi_npt_mdp['nsteps'] = str(int(time / 2e-6))
    if config.MD_settings.Fix:
        equi_npt_mdp.update({'define': '-DFIX'})
        equi_npt_mdp.update({"refcoord_scaling":'com'})
    write_mdp(equi_npt_mdp, r'./equi_npt.mdp')


    #生成拉伸mdp文件
    if config.SMD_settings.run:
        pull_speed = config.SMD_settings.speed
        pull_time = pulling_length / pull_speed
        if config.SMD_settings.time>0:
            pull_time = config.SMD_settings.time
        temp = config.MD_settings.temp
        pull_mdp = copy.deepcopy(PULL_MDP)
        pull_mdp['ref_t'] = f"{temp} {temp}"
        pull_mdp['nsteps'] = str(int(pull_time/2e-6))
        pull_mdp['pull-coord1-rate'] = str(pull_speed/1e3)
        pull_mdp['pull_coord1_k'] = config.SMD_settings.k
        pull_mdp['pull-coord1-type'] = config.SMD_settings.type
        if config.SMD_settings.type == 'constant-force':
            pull_mdp['pull-coord1-rate'] = 0
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
        #prod_mdp["pcoupl"] = 'no' #变温MD模拟使用NVP
        prod_mdp["pcoupl"] = 'Parrinello-Rahman'
    if config.MD_settings.Fix:
        prod_mdp.update({'define': '-DFIX'})
        prod_mdp.update({"refcoord_scaling":'com'})
    
    write_mdp(prod_mdp, r'./md.mdp')

    system("rm *#*")

def mini_run(config, gpu_id=0):
    if Path('./mini.gro').is_file():
        return None
    gmx_exec = config.global_settings.gmx_exec
    cpu_num = config.global_settings.cpu_num
    system(f"{gmx_exec} grompp -f mini.mdp -o mini.tpr -c input.pdb -r input.pdb -p topol.top -n index.ndx")
    wait_for_gpu(gpu_id,sleep_time=20)
    system(f"{gmx_exec} mdrun -v -deffnm mini  -ntmpi 1 -ntomp {cpu_num} -pin on -pinoffset {gpu_id*cpu_num*2} -gpu_id {gpu_id}")

def equi_run(config, gpu_id=0):
    if Path("./equi.gro").is_file():
        return None
    bonded = config.global_settings.gmx_bonded
    update = config.global_settings.gmx_update
    gmx_exec = config.global_settings.gmx_exec
    cpu_num = config.global_settings.cpu_num
    if not Path("equi_nvt.cpt").is_file():
        system(f"{gmx_exec} grompp -f equi_nvt.mdp -o equi_nvt.tpr -c mini.gro -r mini.gro -p topol.top -n index.ndx")
        wait_for_gpu(gpu_id,sleep_time=20)
        system(f"{gmx_exec} mdrun -v -deffnm equi_nvt  -ntmpi 1 -ntomp {cpu_num} -gpu_id {gpu_id} -pin on -pinoffset {gpu_id*cpu_num*2} -update {update}  -notunepme -bonded {bonded}")
    else:
        wait_for_gpu(gpu_id,sleep_time=20)
        system(f"{gmx_exec} mdrun -v -deffnm equi_nvt  -ntmpi 1 -ntomp {cpu_num} -gpu_id {gpu_id} -pin on -pinoffset {gpu_id*cpu_num*2} -update {update}  -notunepme -bonded {bonded} -cpi equi_nvt.cpt")

    if not Path("equi_npt.cpt").is_file():
        system(f"{gmx_exec} grompp -f equi_npt.mdp -o equi_npt.tpr -c equi_nvt.gro -r equi_nvt.gro -p topol.top -n index.ndx")
        wait_for_gpu(gpu_id,sleep_time=20)
        system(f"{gmx_exec} mdrun -v -deffnm equi_npt  -ntmpi 1 -ntomp {cpu_num} -gpu_id {gpu_id} -pin on -pinoffset {gpu_id*cpu_num*2} -update {update}  -notunepme -bonded {bonded}")
    else:
        wait_for_gpu(gpu_id,sleep_time=20)
        system(f"{gmx_exec} mdrun -v -deffnm equi_npt  -ntmpi 1 -ntomp {cpu_num} -gpu_id {gpu_id} -pin on -pinoffset {gpu_id*cpu_num*2} -update {update}  -notunepme -bonded {bonded} -cpi equi_npt.cpt")


def md_run(config, gpu_id=0):
    if not config.MD_settings.run:
        return None
    bonded = config.global_settings.gmx_bonded
    update = config.global_settings.gmx_update
    gmx_exec = config.global_settings.gmx_exec
    cpu_num = config.global_settings.cpu_num
    times = config.MD_settings.times
    start_file = 'equi_npt'

    for i in range(times):
        i = i+1
        name = f"md_{i}"
        if not Path(f"{name}.cpt").is_file():
            system(f"{gmx_exec} grompp -f md.mdp -o {name}.tpr -c {start_file}.gro -r {start_file}.gro -p topol.top -n index.ndx -maxwarn 1")
        run_scheduler = [config.MD_settings.time]
        if config.MD_settings.early_stop:
            md_time = config.MD_settings.time
            early_stop_step = config.MD_settings.early_stop_step
            start_step = early_stop_step
            run_scheduler = list(range(start_step,md_time,early_stop_step))+run_scheduler
        reach_time = max([float(fname.stem.split('_')[-1]) for fname in Path("./").rglob(f"{name}*calc_at*.txt")]+[0])
        run_scheduler = [time for time in run_scheduler if time > reach_time or time <=0]
        for time in run_scheduler:
            if config.MD_settings.early_stop:
                early_stop_rmsd(name,time,config)
                if Path(f"{name}_earlystop").is_file():
                    break
            system(f"{gmx_exec} convert-tpr -s {name}.tpr -o {name}.tpr -until {time*1000}")
            system("rm *#*")
            if Path(f'{name}.cpt').is_file():
                wait_for_gpu(gpu_id,sleep_time=20)
                system(f"{gmx_exec} mdrun -v -deffnm {name}  -ntmpi 1 -ntomp {cpu_num} -gpu_id {gpu_id} -pin on -pinoffset {gpu_id * cpu_num*2} -update {update}  -notunepme -bonded {bonded} -cpi {name}.cpt")
            else:
                wait_for_gpu(gpu_id,sleep_time=20)
                system(f"{gmx_exec} mdrun -v -deffnm {name}  -ntmpi 1 -ntomp {cpu_num} -gpu_id {gpu_id} -pin on -pinoffset {gpu_id * cpu_num*2} -update {update}  -notunepme -bonded {bonded}")
        if config.MD_settings.early_stop:
            early_stop_rmsd(name, time, config)

def smd_cs_run(config, gpu_id=0):
    if not config.SMD_settings.run or config.SMD_settings.type == 'constant-force':
        return None
    bonded = config.global_settings.gmx_bonded
    update = config.global_settings.gmx_update
    gmx_exec = config.global_settings.gmx_exec
    cpu_num = config.global_settings.cpu_num
    times = config.SMD_settings.times
    if Path("md_1.gro").is_file():
        start_file = 'md_1'
    else:
        start_file = 'equi_npt'
    for i in range(times):
        name = f'pull_cs_{i+1}'
        if Path(f"{name}.cpt").is_file():
            wait_for_gpu(gpu_id,sleep_time=20)
            system(f"{gmx_exec} mdrun -v -deffnm {name}  -ntmpi 1 -ntomp {cpu_num} -gpu_id {gpu_id} -pin on -pinoffset {gpu_id * cpu_num*2} -update {update} -notunepme -bonded {bonded} -cpi {name}.cpt -px {name}_pullx.xvg -pf {name}_pullf.xvg")
        else:
            system(f"{gmx_exec} grompp -f pull.mdp -o {name}.tpr -c {start_file}.gro -r {start_file}.gro -p topol.top -n index.ndx")
            wait_for_gpu(gpu_id,sleep_time=20)
            system(f"{gmx_exec} mdrun -v -deffnm {name}  -ntmpi 1 -ntomp {cpu_num} -gpu_id {gpu_id} -pin on -pinoffset {gpu_id * cpu_num*2} -update {update}  -notunepme -bonded {bonded}")
def smd_cf_run(config, gpu_id=0):
    if not config.SMD_settings.run or config.SMD_settings.type == 'umbrella':
        return None
    bonded = config.global_settings.gmx_bonded
    update = config.global_settings.gmx_update
    gmx_exec = config.global_settings.gmx_exec
    cpu_num = config.global_settings.cpu_num
    times = config.SMD_settings.times
    if Path("md_1.gro").is_file():
        start_file = 'md_1'
    else:
        start_file = 'equi_npt'
    for i in range(times):
        name = f'pull_cf_{i+1}'
        curr_time = 0 #ps
        if Path(f"{name}_pullx.xvg").is_file():
            data = np.loadtxt(f"{name}_pullx.xvg",comments=['#',"@"])
            curr_time = data[:,0][-1]
        while True:
            curr_time += 10 * 1000
            if curr_time > config.SMD_settings.time*1e3:
                break
            if not Path(f"{name}.tpr").is_file():
                system(f"{gmx_exec} grompp -f pull.mdp -o {name}.tpr -c {start_file}.gro -r {start_file}.gro -p topol.top -n index.ndx")
            system(f"{gmx_exec} convert-tpr -s {name}.tpr -o {name}.tpr -until {curr_time}")
            if Path(f"{name}.cpt").is_file():
                system(f"{gmx_exec} mdrun -v -deffnm {name}  -ntmpi 1 -ntomp {cpu_num} -gpu_id {gpu_id} -pin on -pinoffset {gpu_id * cpu_num*2} -update {update} -notunepme -bonded {bonded} -cpi {name}.cpt -px {name}_pullx.xvg -pf {name}_pullf.xvg")
            else:
                wait_for_gpu(gpu_id,sleep_time=20)
                system(f"{gmx_exec} mdrun -v -deffnm {name}  -ntmpi 1 -ntomp {cpu_num} -gpu_id {gpu_id} -pin on -pinoffset {gpu_id * cpu_num*2} -update {update}  -notunepme -bonded {bonded}")
            data = np.loadtxt(f"{name}_pullx.xvg",comments=['#',"@"])
            curr_x = data[:,1][-1]-data[:,1][0]
            if curr_x >= config.SMD_settings.pull_distance or data[:,1].min()<0:
                break
def MD(config, fname, gpu_id=0):
    system("rm *#*")
    step1_preparation(config, fname)
    mini_run(config, gpu_id)
    equi_run(config, gpu_id)
    md_run(config, gpu_id)
    smd_cs_run(config, gpu_id)
    smd_cf_run(config, gpu_id)
    system("rm *#*")



@hydra.main(version_base=None, config_path="", config_name="config")
def main(config):
    fname = './protein.pdb'
    MD(config, fname, gpu_id=7)

if __name__ == '__main__':
    main()
