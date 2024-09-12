import shutil
import argparse
from tqdm.auto import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from models.AMG_PTPkt import AMG_PTPkt
from utils.datasets import *
from utils.misc import *
from utils.train import *
from utils.data import *
from utils.mol_tree import *
from utils.transforms import *
from torch.utils.data import DataLoader


def setup_logging(args):
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))
    return logger, writer, ckpt_dir

def create_transforms():
    pocket_featurizer = FeaturizePocketAtom()
    return Compose([pocket_featurizer]), pocket_featurizer

def setup_datasets_loaders(config, transform, logger):
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(config=config.dataset, transform=transform)
    train_set, val_set = subsets['train'], subsets['test']
    train_iterator = inf_iterator(DataLoader(train_set, batch_size=config.train.batch_size,
                                             shuffle=True, num_workers=config.train.num_workers,
                                             collate_fn=collate_pocket))
    val_loader = DataLoader(val_set, batch_size=config.train.batch_size, shuffle=False,
                            num_workers=config.train.num_workers, collate_fn=collate_pocket)
    return train_iterator, val_loader

def setup_model(config, pocket_featurizer, device):
    model = AMG_PTPkt(config.model, pocket_atom_feature_dim=pocket_featurizer.feature_dim).to(device)
    return model

def main(args):
    # Load configs
    config = load_config(args.config)
    seed_all(config.train.seed)

    # Setup logging
    logger, writer, ckpt_dir = setup_logging(args)
    logger.info(args)
    logger.info(config)

    # Transforms, datasets, and loaders
    transform, pocket_featurizer = create_transforms()
    train_iterator, val_loader = setup_datasets_loaders(config, transform, logger)

    # Model
    logger.info('Building model...')
    model = setup_model(config, pocket_featurizer, args.device)

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

        loss = model.get_loss(
            pocket_pos=batch['pocket_pos'],
            pocket_atom_feature=batch['pocket_atom_feature'].float(),
            batch_pocket=batch['pocket_element_batch'],
            batch=batch)
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        logger.info('[Train] Iter %d | Loss %.6f | Orig_grad_norm %.6f' % (it, loss.item(), orig_grad_norm))
        writer.add_scalar('train/loss', loss, it)
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
                loss = model.get_loss(
                    pocket_pos=batch['pocket_pos'],
                    pocket_atom_feature=batch['pocket_atom_feature'].float(),
                    batch_pocket=batch['pocket_element_batch'],
                    batch=batch)
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

    # Training loop
    try:
        for it in range(1, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                validate(it)
                ckpt_path = os.path.join(ckpt_dir, f'{it}.pt')
                torch.save({
                    'config': config,
                    'pocket_atom_emb': model.pocket_atom_emb.state_dict(),
                    'pocket_encoder': model.pocket_encoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/pretrain_pocket.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()

    main(args)