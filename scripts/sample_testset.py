from models.bio_markov import CustomEnv
from spinningup.spinup.utils.mpi_tools import mpi_fork
from spinningup.spinup.algos.pytorch.ppo.ppo import PPOBuffer
import gym
from models.actor_critic import AMG_ActorCritic
from utils.misc import load_config, get_new_log_dir, get_logger
from utils.mol_tree import Vocab
import time
import numpy as np
import torch
import shutil
from torch.optim import Adam
import gym
import time
from spinningup.spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params
from spinningup.spinup.utils.mpi_tools import mpi_fork, num_procs
import argparse
from models.AMG import AMG
from utils.datasets import *
from utils.misc import *
from utils.train import *
from utils.data import *
from utils.mol_tree import *
from utils.transforms import *
from utils.metric import *
from torch.utils.data import DataLoader
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--outdir', type=str, default='./sampled_results/')
    parser.add_argument('--config', type=str, default='./configs/rl.yml')
    parser.add_argument('--load_rl_model', type=str, default='./ckpts/rl_agent.pt')
    parser.add_argument('--PR_path', type=str, default='./ckpts/pr_model.pt',
                            help='Load pretraining representation of ligand and pocket') 
    parser.add_argument('--temp_dir', type=str, default='./temp_files/')
    parser.add_argument('--vocab_path', type=str, default='./dataset/vocab_np_crossdocked_pocket.txt')
    parser.add_argument('--start_index', type=str, default=0)
    parser.add_argument('--end_index', type=str, default=99)
    args = parser.parse_args()
   
    # Load config and vocab
    config = load_config(args.config)
    
    # Random seed
    seed = config.train.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    vocab = []
    for line in open(args.vocab_path):
        p1, _, p3 = line.partition(':')
        vocab.append(p1)
    vocab = Vocab(vocab)

    # Transforms
    pocket_featurizer = FeaturizePocketAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    masking = LigandMaskAll(vocab)
    transform = Compose([
        LigandCountNeighbors(),
        pocket_featurizer,
        ligand_featurizer,
        FeaturizeLigandBond(),
        masking,
    ])

    # Datasets and loaders
    _, subsets = get_dataset(config=config.dataset, transform=transform)
    data_ids = range(args.start_index, args.end_index+1)
    for data_id in data_ids:
        data = subsets['test'][data_id]
        test_set = [data for _ in range(1)]
        test_iterator = inf_iterator(DataLoader(test_set, batch_size=1, 
                                                shuffle=True, num_workers=config.train.num_workers,
                                                collate_fn=collate_pocket_ligand, drop_last=True))

        if args.PR_path:
            ckpt = torch.load(args.PR_path, map_location=args.device)
        model = AMG(
            ckpt['config'].model,
            pocket_atom_feature_dim=pocket_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim,
            vocab=vocab,
            device=args.device
        ).to(args.device)
        if args.PR_path:
            model.load_state_dict(ckpt['model'])

        with torch.no_grad():
            model.eval()

        # Logging
        log_dir = get_new_log_dir(args.outdir, prefix='%d-%s' % (data_id, data['pocket_filename'].split('/')[0]))
        logger = get_logger('sample', log_dir)
        logger.info(args)
        logger.info(config)
        shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

        with open(os.path.join(log_dir, 'pocket_info.txt'), 'a') as f:
            f.write(data['pocket_filename'] + '\n')

        mpi_fork(args.cpu)  # run parallel code with mpi

        env_sys = gym.make(config.RL.env)
        env_cus = CustomEnv(config, vocab, model, args, mode='eval', logdir=log_dir)

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

        # Instantiate environment
        env = env_cus
        obs_dim = env.observation_space.shape

        act_dim = 1

        # Create actor-critic module
        ac = AMG_ActorCritic(env.observation_space, env.action_space, hidden_sizes=[config.RL.hid]*config.RL.l, device=args.device)  
        ac_ckpt = torch.load(args.load_rl_model, map_location=args.device)
        ac.load_state_dict(ac_ckpt['ac'])
        with torch.no_grad():
            ac.eval()
        
        # Sync params across processes
        sync_params(ac)

        # Set up experience buffer
        local_steps_per_epoch = int(config.sample.steps / num_procs())

        buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, config.RL.gamma, config.RL.lam)
        
        # Set up optimizers for policy and value function
        pi_optimizer = Adam(ac.pi.parameters(), lr=config.RL.pi_lr)
        vf_optimizer = Adam(ac.v.parameters(), lr=config.RL.vf_lr)

        # Prepare for interaction with environment
        start_time = time.time()
        
        o, info, ep_ret, ep_len = env.reset(next(test_iterator)), {}, 0, 0
 
        data_list = []

        while True:
            for t in range(local_steps_per_epoch):
                a, v, logp = ac.step(o)
                next_o, r, d = env.step(a)
                ep_len += 1
                if r:
                    if r.GetNumAtoms() >= 8:
                        try:
                            mol_h = r
                            mol_h = Chem.AddHs(mol_h, addCoords=True)
                            UFFOptimizeMolecule(mol_h)
                            data_list.append(r)
                        except:
                            pass
            
                # Update obs (critical!)
                o = next_o
                if d:
                    print(len(data_list))
                    o, info, ep_ret, ep_len = env.reset(next(test_iterator)), {}, 0, 0     
            if len(data_list) >= config.sample.num_samples:
                break
            torch.cuda.empty_cache()
            
        with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
            for i, mol in enumerate(data_list):
                smiles_f.write(Chem.MolToSmiles(mol) + '\n')
                Chem.MolToMolFile(mol, os.path.join(log_dir, '%d.sdf' % i))
