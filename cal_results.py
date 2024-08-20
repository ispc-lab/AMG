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


def cal_metrics(mol_list, pdb_path, save_path):
    qed_list, logp_list, sa_list, Lip_list, qvina_list = [], [], [], []
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
    
    df = pd.DataFrame({'SMILES': [Chem.MolToSmiles(i) for i in rdmol_list], 'Qvina': qvina_list, 'QED': qed_list, 'LogP': logp_list, 
                       'SA': sa_list, 'Lip': Lip_list})
    df.to_csv(save_path + '.csv', index=False, mode='a')


def gen_results_file(path, save_result_path):  
    if not os.path.exists(save_result_path):
        os.makedirs(save_result_path)
    for result_dir in os.listdir(path):

        mol_list = []
        for file_name in os.listdir(path + result_dir):
            if file_name == 'pocket_info.txt':
                pdb_file_path = path + result_dir + '/' + file_name
                with open(pdb_file_path, 'r') as file:
                    pdb_name = file.readline()
                pdb_path = os.path.join(config.dataset.path, pdb_name)
            if file_name[-3:] == 'sdf':
                sdf_dir = path + result_dir + '/' + file_name
                mol_list.append(Chem.MolFromMolFile(sdf_dir))
    
        cal_metrics(mol_list, pdb_path, save_result_path + result_dir)





if __name__ == '__main__':
    # Load configs
    config = load_config('./configs/sample.yml')
    seed_all(config.sample.seed)

    path = 'results/new/RL_np_reward/sample_1200/'  # sdf files 
    save_result_path = 'results/new/RL_np_reward/sample_1200_results/'
    gen_results_file(path, save_result_path)

# results_path = '/DATA1/east/ResGen-main/outputs_final_results/'
# results_path = '/DATA1/east/targetdiff-main/results_100/'
# results_path = '/DATA1/east/DecompDiff-main/results/'

# results_path = 'results/new/RL_np_reward_recovvery_ligand&protein/ckpt_200_results/'
# results_path = '/home/east/projects/ScaRLPR/results/new/np_new_dataset/results_-5-5-5/'
# results_path = 'results/ablation/3_wo_RL&PR_results/'
# results_path = 'results/ablation/wo_PR_results/'
# results_path = 'results/ablation/1_woRL_66k_results/'
# results_path = 'results/obs60/ckpt200_results/'
