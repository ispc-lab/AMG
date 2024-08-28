import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import pandas as pd
from rdkit import Chem
from utils.docking import cal_docking
from utils.metric import *
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from rdkit.Chem.Descriptors import qed
from utils.sascorer import compute_sa_score
from utils.misc import *

receptor_path = "/DATA1/east/AMG/ADFRsuite_x86_64Linux_1.0/myFolder/bin"
vina_path = "/DATA1/east/AMG/Qvina/bin"
os.environ['PATH'] += os.pathsep + receptor_path
os.environ['PATH'] += os.pathsep + vina_path


def cal_metrics(mol_list, pdb_path):
    qed_list, sa_list, Lip_list, qvina_list = [], [], [], []
    rdmol_list = []
    for mol in mol_list:
        try:
            Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            rdmol_list.append(mol)
        except:
            pass

    for mol in rdmol_list:
        qed_list.append(qed(mol))
        sa_list.append(compute_sa_score(mol))
        Lip_list.append(lipinski(mol))

        try:
            ligand_rdmol = Chem.AddHs(mol, addCoords=True)
            UFFOptimizeMolecule(ligand_rdmol)
            qvina_list.append(cal_docking(ligand_rdmol, pdb_path))
        except:
            qvina_list.append('nan')
    
    df = pd.DataFrame({'SMILES': [Chem.MolToSmiles(i) for i in rdmol_list], 'Qvina': qvina_list, 'QED': qed_list, 
                       'SA': sa_list, 'Lip': Lip_list})
    return df


def gen_results_file(path, save_result_path):  
    eval_results = {}

    for result_dir in tqdm(os.listdir(path)):
        mol_list = []
        result_dir_path = os.path.join(path, result_dir)
        for file_name in os.listdir(result_dir_path):
            file_path = os.path.join(result_dir_path, file_name)
            if file_name == 'pocket_info.txt': 
                with open(file_path, 'r') as file:
                    pdb_name = file.readline()
                pdb_path = os.path.join(config.dataset.path, pdb_name)
            if file_name[-3:] == 'sdf':
                mol_list.append(Chem.MolFromMolFile(file_path))
        eval_results[result_dir] = cal_metrics(mol_list, pdb_path)
        
    torch.save(eval_results, save_result_path)


if __name__ == '__main__':
    # Load configs
    config = load_config('./configs/sample.yml')
    seed_all(config.sample.seed)

    path = 'eval_results/baselines_results/org_sdf_files/AMG/'  # input sdf path 
    save_result_path = 'results/eval_files.pt'
    gen_results_file(path, save_result_path)
