import warnings
warnings.filterwarnings("ignore")
import shutil
import argparse
from tqdm.auto import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from models.AMG_PTLig import AMG_PTLig
from utils.datasets import get_dataset
from utils.misc import *
from utils.train import *
from utils.data import collate_ligand
from utils.transforms import *
from utils.mol_tree import *
from torch.utils.data import DataLoader


def load_vocab(vocab_path):
    vocab = []
    with open(vocab_path) as f:
        for line in f:
            p, _, _ = line.partition(':')
            vocab.append(p)
    return Vocab(vocab)


def setup_logging(config, args):
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
    ligand_featurizer = FeaturizeLigandAtom()
    masking = get_mask(config.train.transform.mask, vocab)
    return Compose([
        LigandCountNeighbors(),
        ligand_featurizer,
        FeaturizeLigandBond(),
        masking,
    ]), ligand_featurizer


def setup_datasets_loaders(config, transform, logger):
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(config=config.dataset, transform=transform)
    train_set, val_set = subsets['train'], subsets['test']
    train_iterator = inf_iterator(DataLoader(train_set, batch_size=config.train.batch_size,
                                             shuffle=True, num_workers=config.train.num_workers,
                                             collate_fn=collate_ligand))
    val_loader = DataLoader(val_set, batch_size=config.train.batch_size, shuffle=False,
                            num_workers=config.train.num_workers, collate_fn=collate_ligand, drop_last=True)
    return train_iterator, val_loader


def setup_model(config, ligand_featurizer, vocab, device):
    model = AMG_PTLig(
        config.model,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        vocab=vocab).to(device)
    return model


def save_checkpoint(state, ckpt_dir, iteration):
    ckpt_path = os.path.join(ckpt_dir, f'{iteration}.pt')
    torch.save(state, ckpt_path)


def main(args):
    # Load vocab and configs
    vocab = load_vocab(args.vocab_path)
    config = load_config(args.config)
    seed_all(config.train.seed)

    # Setup logging
    logger, writer, ckpt_dir = setup_logging(config, args)
    logger.info(args)
    logger.info(config)

    # Transforms, datasets, and loaders
    transform, ligand_featurizer = create_transforms(config, vocab)
    train_iterator, val_loader = setup_datasets_loaders(config, transform, logger)

    # Model
    logger.info('Building model...')
    model = setup_model(config, ligand_featurizer, vocab, args.device)

    # Optimizer and scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    # Training and validation functions
    def train(it):
        model.train()
        optimizer.zero_grad()
        batch = next(train_iterator)
        for key in batch:
            batch[key] = batch[key].to(args.device)

        loss, loss_list = model.get_loss(
            ligand_pos=batch['ligand_pos'],
            ligand_atom_feature=batch['ligand_atom_feature_full'].float(),
            batch_ligand=batch['ligand_element_batch'],
            ligand_pos_torsion=batch['ligand_pos_torsion'],
            ligand_atom_feature_torsion=batch['ligand_feature_torsion'].float(),
            batch_ligand_torsion=batch['ligand_element_torsion_batch'],
            batch = batch
            )
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        logger.info('[Train] Iter %d | Loss %.6f | Loss(SSL) %.6f | Loss(Pred) %.6f | Loss(comb) %.6f | Loss(Focal) %.6f | Loss(Tor) %.6f | Orig_grad_norm %.6f' % (it, loss.item(), loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[4], orig_grad_norm))
        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('train/SSL', loss_list[0], it)
        writer.add_scalar('train/pred_loss', loss_list[1], it)
        writer.add_scalar('train/comb_loss', loss_list[2], it)
        writer.add_scalar('train/focal_loss', loss_list[3], it)
        writer.add_scalar('train/torsion_loss', loss_list[4], it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad', orig_grad_norm, it)
        writer.flush()

    def validate(it):
        sum_loss, sum_n = 0, 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                for key in batch:
                    batch[key] = batch[key].to(args.device)
                loss, _ = model.get_loss(
                    ligand_pos=batch['ligand_pos'],
                    ligand_atom_feature=batch['ligand_atom_feature_full'].float(),
                    batch_ligand=batch['ligand_element_batch'],
                    ligand_pos_torsion=batch['ligand_pos_torsion'],
                    ligand_atom_feature_torsion=batch['ligand_feature_torsion'].float(),
                    batch_ligand_torsion=batch['ligand_element_torsion_batch'],
                    batch = batch
                    )
            
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

    try:
        for it in range(1, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                validate(it)
                save_checkpoint({
                    # State dict and other information to save
                }, ckpt_dir, it)
    except KeyboardInterrupt:
        logger.info('Terminating...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/pretrain_ligand.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./logs/PretrainLigand/')
    parser.add_argument('--vocab_path', type=str, default='./dataset/vocab_pretrain.txt')
    parser.add_argument('--start_iter', type=int, default=0)
    args = parser.parse_args()

    main(args)

