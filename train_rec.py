import warnings
warnings.filterwarnings("ignore")
import shutil
import argparse
from tqdm.auto import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from models.AMG import AMG
from utils.datasets import *
from utils.misc import *
from utils.train import *
from utils.data import *
from utils.mol_tree import *
from utils.transforms import *
from utils.metric import *
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

def load_vocab(vocab_path):
    vocab = []
    with open(vocab_path) as f:
        for line in f:
            p, _, _ = line.partition(':')
            vocab.append(p)
    return Vocab(vocab)

def setup_logging(args, config):
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))
    return logger, writer, ckpt_dir

def create_transforms(config, vocab):
    pocket_featurizer = FeaturizePocketAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    masking = get_mask(config.train.transform.mask, vocab)
    return Compose([
        LigandCountNeighbors(),
        pocket_featurizer,
        ligand_featurizer,
        FeaturizeLigandBond(),
        masking,
    ]), pocket_featurizer, ligand_featurizer

def setup_datasets_loaders(config, transform, logger):
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(config=config.dataset, transform=transform)
    train_set, val_set = subsets['train'], subsets['test']
    train_iterator = inf_iterator(DataLoader(train_set, batch_size=config.train.batch_size,
                                             shuffle=True, num_workers=config.train.num_workers,
                                             collate_fn=collate_pocket_ligand))
    val_loader = DataLoader(val_set, batch_size=config.train.batch_size, shuffle=False,
                            num_workers=config.train.num_workers, collate_fn=collate_pocket_ligand, drop_last=True)
    return train_iterator, val_loader

def setup_model(config, pocket_featurizer, ligand_featurizer, vocab, args):
    model = AMG(
        config.model,
        pocket_atom_feature_dim=pocket_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        vocab=vocab,
        ckpt_ligand_path=args.PR_ligand_path,
        ckpt_pocket_path=args.PR_pocket_path,
        device=args.device).to(args.device)
    if args.PR_path:
        ckpt = torch.load(args.PR_path, map_location=args.device)
        model.load_state_dict(ckpt['model'])
    return model


def train(model, optimizer, train_iterator, logger, writer, args, config, it):
    model.train()
    optimizer.zero_grad()
    
    batch = next(train_iterator)

    for key in batch:
        if key not in ['pocket_filename', 'ligand_filename']:
            batch[key] = batch[key].to(args.device)

    loss, loss_list = model.get_loss(batch)

    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
    optimizer.step()

    logger.info('[Train] Iter %d | Loss %.6f | Loss(Interaction) %.6f | Loss(Pred) %.6f | Loss(comb) %.6f | Loss(Focal) %.6f | Loss(Dm) %.6f '
                '| Loss(Tor) %.6f | Loss(Recovery) %.6f  | Orig_grad_norm %.6f' % (it, loss.item(), loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[4], loss_list[5], loss_list[6], orig_grad_norm))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/interaction_loss', loss_list[0], it)
    writer.add_scalar('train/pred_loss', loss_list[1], it)
    writer.add_scalar('train/comb_loss', loss_list[2], it)
    writer.add_scalar('train/focal_loss', loss_list[3], it)
    writer.add_scalar('train/dm_loss', loss_list[4], it)
    writer.add_scalar('train/torsion_loss', loss_list[5], it)
    writer.add_scalar('train/recovery_loss', loss_list[6], it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad', orig_grad_norm, it)
    writer.flush()


def validate(model, val_loader, logger, writer, scheduler, config, it):
    sum_loss, sum_n = 0, 0
    with torch.no_grad():
        model.eval()
        for batch in tqdm(val_loader, desc='Validate'):
            for key in batch:
                if key not in ['pocket_filename', 'ligand_filename']:
                    batch[key] = batch[key].to(args.device)
            loss, _ = model.get_loss(batch)
            sum_loss += loss.item()
            sum_n += 1
    avg_loss = sum_loss / sum_n

    if config.train.scheduler.type == 'plateau':
        scheduler.step(avg_loss)
    elif config.train.scheduler.type == 'warmup_plateau':
        scheduler.step_ReduceLROnPlateau(avg_loss)
    else:
        scheduler.step()

    logger.info('[Validate] Iter %05d | Loss %.6f' % (it, avg_loss,))
    writer.add_scalar('val/loss', avg_loss, it)
    writer.flush()
    return avg_loss

def main(args):
    # Load vocab and configs
    vocab = load_vocab(args.vocab_path)
    config = load_config(args.config)
    seed_all(config.train.seed)

    # Setup logging
    logger, writer, ckpt_dir = setup_logging(args, config)
    logger.info(args)
    logger.info(config)

    # Transforms, datasets, and loaders
    transform, pocket_featurizer, ligand_featurizer = create_transforms(config, vocab)
    train_iterator, val_loader = setup_datasets_loaders(config, transform, logger)

    # Model
    logger.info('Building model...')
    model = setup_model(config, pocket_featurizer, ligand_featurizer, vocab, args)

    # Optimizer and scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    # Training loop
    for it in range(1, config.train.max_iters + 1):
        train(model, optimizer, train_iterator, logger, writer, args, config, it)
        if it % config.train.val_freq == 0 or it == config.train.max_iters:
            validate(model, val_loader, logger, writer, scheduler, config, it)
            # Save checkpoint
            ckpt_path = os.path.join(ckpt_dir, f'{it}.pt')
            torch.save({
                'config': config,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iteration': it,
            }, ckpt_path)
            torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train_model.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./logs/new_dataset_AMG/recovery/')
    parser.add_argument('--temp_dir', type=str, default='/home/east/projects/Sca/temp_files')
    parser.add_argument('--vocab_path', type=str, default='dataset/vocab_pretrain.txt')
    parser.add_argument('--PR_path', type=str, default=None, help='Load pretraining representation of ligand and pocket')
    parser.add_argument('--PR_ligand_path', type=str, default='logs/PretrainLigand/vocab_np/pretrain_ligand_2023_10_26__00_05_27/checkpoints/1000000.pt', help='Load pretraining representation of ligand')
    parser.add_argument('--PR_pocket_path', type=str, default='logs/pocket_pretrain_logs/pretrain_pocket_2023_10_26__10_04_48/checkpoints/1000000.pt', help='Load pretraining representation of pocket')
    args = parser.parse_args()

    main(args)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, default='./configs/train_model.yml')
#     parser.add_argument('--device', type=str, default='cuda:0')
#     parser.add_argument('--logdir', type=str, default='./logs/new_dataset_AMG/recovery/')
#     parser.add_argument('--temp_dir', type=str, default='/home/east/projects/Sca/temp_files')
#     parser.add_argument('--vocab_path', type=str, default='dataset/vocab_pretrain.txt')
#     parser.add_argument('--PR_path', type=str, default=None, help='Load pretraining representation of ligand and pocket')                 # '/media/data2/xudong/logs/fine_tuning_logs/train_model_2023_09_10__20_19_52/checkpoints/920000.pt')          # None or load path
#     parser.add_argument('--PR_ligand_path', type=str, default='logs/PretrainLigand/vocab_np/pretrain_ligand_2023_10_26__00_05_27/checkpoints/1000000.pt', help='Load pretraining representation of ligand')              # '/media/data2/xudong/logs/ligand_pretrain_logs/train_model_2023_09_14__19_00_44/checkpoints/100000.pt')          # None or load path
#     parser.add_argument('--PR_pocket_path', type=str, default='logs/pocket_pretrain_logs/pretrain_pocket_2023_10_26__10_04_48/checkpoints/1000000.pt', help='Load pretraining representation of pocket')          # None or load path
#     args = parser.parse_args()

#     # Load vocab
#     vocab = []
#     for line in open(args.vocab_path):
#         p, _, _ = line.partition(':')
#         vocab.append(p)
#     vocab = Vocab(vocab)

#     # Load configs
#     config = load_config(args.config)
#     config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
#     seed_all(config.train.seed)

#     # Logging
#     log_dir = get_new_log_dir(args.logdir, prefix=config_name)
#     ckpt_dir = os.path.join(log_dir, 'checkpoints')
#     os.makedirs(ckpt_dir, exist_ok=True)
#     logger = get_logger('train', log_dir)
#     writer = torch.utils.tensorboard.SummaryWriter(log_dir)
#     logger.info(args)
#     logger.info(config)
#     shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
#     shutil.copytree('./models', os.path.join(log_dir, 'models'))

#     # Transforms
#     pocket_featurizer = FeaturizePocketAtom()
#     ligand_featurizer = FeaturizeLigandAtom()
#     masking = get_mask(config.train.transform.mask, vocab)
#     transform = Compose([
#         LigandCountNeighbors(),
#         pocket_featurizer,
#         ligand_featurizer,
#         FeaturizeLigandBond(),
#         masking,
#     ])

#     # Datasets and loaders
#     logger.info('Loading dataset...')
#     dataset, subsets = get_dataset(config=config.dataset, transform=transform)
#     train_set, val_set = subsets['train'], subsets['test']
#     train_iterator = inf_iterator(DataLoader(train_set, batch_size=config.train.batch_size,
#                                              shuffle=True, num_workers=config.train.num_workers,
#                                              collate_fn=collate_pocket_ligand))
#     val_loader = DataLoader(val_set, batch_size=config.train.batch_size, shuffle=False,
#                             num_workers=config.train.num_workers, collate_fn=collate_pocket_ligand, drop_last=True)

#     # Model
#     logger.info('Building model...')
#     if args.PR_path:
#         ckpt = torch.load(args.PR_path, map_location=args.device)
#     model = AMG(
#         config.model,
#         pocket_atom_feature_dim=pocket_featurizer.feature_dim,
#         ligand_atom_feature_dim=ligand_featurizer.feature_dim,
#         vocab=vocab,
#         ckpt_ligand_path=args.PR_ligand_path,
#         ckpt_pocket_path=args.PR_pocket_path,
#         device=args.device).to(args.device)
#     if args.PR_path:
#         model.load_state_dict(ckpt['model'])

#     # Optimizer and scheduler
#     optimizer = get_optimizer(config.train.optimizer, model)
#     scheduler = get_scheduler(config.train.scheduler, optimizer)


#     def train(it):
#         model.train()
#         optimizer.zero_grad()
        
#         batch = next(train_iterator)

#         for key in batch:
#             if key not in ['pocket_filename', 'ligand_filename']:
#                 batch[key] = batch[key].to(args.device)

#         loss, loss_list = model.get_loss(batch)

#         loss.backward()
#         orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
#         optimizer.step()

#         logger.info('[Train] Iter %d | Loss %.6f | Loss(Interaction) %.6f | Loss(Pred) %.6f | Loss(comb) %.6f | Loss(Focal) %.6f | Loss(Dm) %.6f '
#                     '| Loss(Tor) %.6f | Loss(Recovery) %.6f  | Orig_grad_norm %.6f' % (it, loss.item(), loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[4], loss_list[5], loss_list[6], orig_grad_norm))
#         writer.add_scalar('train/loss', loss, it)
#         writer.add_scalar('train/interaction_loss', loss_list[0], it)
#         writer.add_scalar('train/pred_loss', loss_list[1], it)
#         writer.add_scalar('train/comb_loss', loss_list[2], it)
#         writer.add_scalar('train/focal_loss', loss_list[3], it)
#         writer.add_scalar('train/dm_loss', loss_list[4], it)
#         writer.add_scalar('train/torsion_loss', loss_list[5], it)
#         writer.add_scalar('train/recovery_loss', loss_list[6], it)
#         writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
#         writer.add_scalar('train/grad', orig_grad_norm, it)
#         writer.flush()


#     def validate(it):
#         sum_loss, sum_n = 0, 0
#         with torch.no_grad():
#             model.eval()
#             for batch in tqdm(val_loader, desc='Validate'):
#                 for key in batch:
#                     if key not in ['pocket_filename', 'ligand_filename']:
#                         batch[key] = batch[key].to(args.device)
#                 loss, _ = model.get_loss(batch)
#                 sum_loss += loss.item()
#                 sum_n += 1
#         avg_loss = sum_loss / sum_n

#         if config.train.scheduler.type == 'plateau':
#             scheduler.step(avg_loss)
#         elif config.train.scheduler.type == 'warmup_plateau':
#             scheduler.step_ReduceLROnPlateau(avg_loss)
#         else:
#             scheduler.step()

#         logger.info('[Validate] Iter %05d | Loss %.6f' % (it, avg_loss,))
#         writer.add_scalar('val/loss', avg_loss, it)
#         writer.flush()
#         return avg_loss

#     for it in range(1, config.train.max_iters + 1):
#     # for it in range(1000000, config.train.max_iters + 1000001):
#         train(it)
#         if it % config.train.val_freq == 0 or it == config.train.max_iters:
#             validate(it)
#             ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
#             torch.save({
#                 'config': config,
#                 'model': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'scheduler': scheduler.state_dict(),
#                 'iteration': it,
#             }, ckpt_path)
#             torch.cuda.empty_cache()
