from models.bio_markov import CustomEnv
from spinningup.spinup.utils.mpi_tools import mpi_fork
from spinningup.spinup.algos.pytorch.ppo.ppo import PPOBuffer
import spinningup.spinup.algos.pytorch.ppo.core as core
from spinup.utils.run_utils import setup_logger_kwargs
import gym
from models.actor_critic import AMG_ActorCritic
from utils.misc import seed_all, load_config, get_new_log_dir, get_logger
from utils.mol_tree import Vocab
import time
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinningup.spinup.algos.pytorch.ppo.core as core
from spinningup.spinup.utils.logx import EpochLogger
from spinningup.spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinningup.spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import torch.utils.tensorboard
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


def compute_loss_pi(data):
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
    
    # Policy loss
    pi, logp = ac.pi(obs, act=act)
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1-config.RL.clip_ratio, 1+config.RL.clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()
    clipped = ratio.gt(1+config.RL.clip_ratio) | ratio.lt(1-config.RL.clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

    return loss_pi, pi_info

def compute_loss_v(data):
    obs, ret = data['obs'], data['ret']
    return ((ac.v(obs) - ret)**2).mean()

def update():
    data = buf.get()
    pi_l_old, pi_info_old = compute_loss_pi(data)
    pi_l_old = pi_l_old.item()
    v_l_old = compute_loss_v(data).item()

    # Train policy with multiple steps of gradient descent
    for i in range(config.RL.train_pi_iters):
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        kl = mpi_avg(pi_info['kl'])
        if kl > 1.5 * config.RL.target_kl:
            logger.log('Early stopping at step %d due to reaching max kl.'%i)
            break
        loss_pi.backward()
        mpi_avg_grads(ac.pi)    # average grads across MPI processes
        pi_optimizer.step()

    logger.store(StopIter=i)

    # Value function learning
    for i in range(config.RL.train_v_iters):
        vf_optimizer.zero_grad()
        loss_v = compute_loss_v(data)
        loss_v.backward()
        mpi_avg_grads(ac.v)    # average grads across MPI processes
        vf_optimizer.step()

    # Log changes from update
    kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
    logger.store(LossPi=pi_l_old, LossV=v_l_old,
                    KL=kl, Entropy=ent, ClipFrac=cf,
                    DeltaLossPi=(loss_pi.item() - pi_l_old),
                    DeltaLossV=(loss_v.item() - v_l_old))

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--log_dir', type=str, default='/DATA1/east/logs/obs_60/')
    parser.add_argument('--config', type=str, default='./configs/rl.yml')
    parser.add_argument('--PR_path', type=str, default='logs/new_dataset_ScaRLPR/recovery/train_model_2023_12_26__19_50_49/checkpoints/680000.pt', help='Load pretraining representation of ligand and pocket') 
    # parser.add_argument('--PR_path', type=str, default='/DATA1/east/ablation/withoutRL_withoutPR_withinteraction_recovery_ligand&pocket/train_model_2024_01_04__12_48_47/checkpoints/940000.pt', help='Load pretraining representation of ligand and pocket') 
    
    parser.add_argument('--temp_dir', type=str, default='/home/east/projects/Sca/temp_files')
    # parser.add_argument('--vocab_path', type=str, default='./dataset/vocab_np_crossdocked_pocket.txt')
    parser.add_argument('--vocab_path', type=str, default='dataset/vocab_pretrain.txt')
    # parser.add_argument('--mode', type=str, default='sample')  # sample
    # parser.add_argument('--trained_env', type=str, default='logs/RL_logs/model_199.pt')  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
    # masking = get_mask(config.train.transform.mask, vocab)
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
    train_set = subsets['train']
    # train_set = [train_set for _ in range(config.sample.num_samples)]
    train_iterator = inf_iterator(DataLoader(train_set, batch_size=1,  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

    mpi_fork(args.cpu)  # run parallel code with mpi
    # logger_kwargs = setup_logger_kwargs(args.log_dir)

    env_sys = gym.make(config.RL.env)
    env_cus = CustomEnv(config, vocab, model, args)

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    # logger = EpochLogger(**logger_kwargs)
    logger = EpochLogger(**{'output_dir': args.log_dir, 'exp_name': 'ppo'})
    # logger.save_config(locals())

    # Instantiate environment
    env = env_cus
    obs_dim = env.observation_space.shape

    act_dim = 1

    # Create actor-critic module
    ac = AMG_ActorCritic(env.observation_space, env.action_space, hidden_sizes=[config.RL.hid]*config.RL.l, device=args.device)
    # if args.load_rl_model:
    #     ac_ckpt = torch.load(args.load_rl_model, map_location=args.device)
    #     ac.load_state_dict(ac_ckpt['ac'])
    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(config.RL.steps / num_procs())
  
    # local_steps_per_epoch = 15
    # buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, config.RL.gamma, config.RL.lam)
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, config.RL.gamma, config.RL.lam)
    
    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=config.RL.pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=config.RL.vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    # Prepare for interaction with environment
    start_time = time.time()
    
    o, info, ep_ret, ep_len = env.reset(next(train_iterator)), {}, 0, 0
    # o = env.reset(next(train_iterator))
    # ep_ret, ep_len = 0, 0
    
    # Main loop: collect experience in env and update/log each epoch
    for epoch in tqdm(range(config.RL.epochs)):
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(o)
            next_o, r, d = env.step(a)
            ep_ret += r
            ep_len += 1
            # save and log
            buf.store(o.cpu().detach().numpy(), a, r, v, logp)     #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            logger.store(VVals=v)
            
            # Update obs (critical!)
            o = next_o
            timeout = ep_len == config.RL.max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                    # v = torch.as_tensor(v).reshape(config.RL.batch_size,)
                    # print("v: " + str(v))
            
                else:
                    v = 0
                    # v = torch.as_tensor(np.zeros((config.RL.batch_size,)), dtype=torch.float32)
            
                buf.finish_path(v)
                if terminal: 
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    
                o, info, ep_ret, ep_len = env.reset(next(train_iterator)), {}, 0, 0     
                
        # Save model
        if (epoch % config.RL.save_freq == 0) or (epoch == config.RL.epochs-1):
            logger.save_state({'env': env}, epoch)
            torch.save({
                'config': config,
                'ac': ac.state_dict(),
            }, args.log_dir + 'model_' + str(epoch) + '.pt')

        # Perform PPO update!
        update()

        
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*config.RL.steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
        
        torch.cuda.empty_cache()
