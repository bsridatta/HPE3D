import os

from torch.cuda import device_count

from trainer_pl import VAEGAN
import pytorch_lightning as pl
from argparse import ArgumentParser
import torch
from dataset import H36M
from pytorch_lightning.loggers import WandbLogger


def main():
    parser = get_argparser()
    opt = parser.parse_args()
    pl.seed_everything(opt.seed)

    train_loader = torch.utils.data.DataLoader(
        H36M(opt.train_file, opt.is_ss, is_train=True, debug=opt.fast_dev_run),
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        H36M(opt.test_file, opt.is_ss, is_train=False, debug=opt.fast_dev_run),
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        shuffle=False,
    )

    logger = WandbLogger(project="gan", reinit=True)
    logger.log_hyperparams(opt)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="mpjpe",
        dirpath="./ckpts/",
        filename=f"{logger.experiment.name}_"+"{epoch}-{mpjpe:.2f}",
        mode="max",
        verbose=True,
    )
    checkpoint_callback.FILE_EXTENSION = '.pt'
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count() * int(opt.gpu),
        fast_dev_run=opt.fast_dev_run,
        max_epochs=opt.epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
    )
    model = VAEGAN(opt)
    trainer.fit(model, train_loader, val_loader)
    print(
        f"Best model @ {trainer.checkpoint_callback.best_model_score}: {trainer.checkpoint_callback.best_model_path}"
    )


def get_argparser():
    parser = ArgumentParser()

    # fmt: off
    # training specific
    parser.add_argument('--epochs', default=2, type=int,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='number of samples per step, have more than one for batch norm')
    parser.add_argument('--fast_dev_run', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='run all methods once to check integrity')
    parser.add_argument('--is_ss', default=True, type=bool,
                        help='training strategy - self supervised')
    # model specific
    parser.add_argument('--latent_dim', default=51, type=int,
                        help='dimensions of the cross model latent space')
    parser.add_argument('--lambda_g', default=1, type=float,
                        help='ss- weight for gen loss/adversarial loss based on disc')
    parser.add_argument('--lambda_recon', default=1e-3, type=float,
                        help='ss - weight for recon loss')
    parser.add_argument('--lambda_kld', default=1e-3, type=float,  # 0.01
                        help='ss - beta (kld coeff in b-vae) or maximum value of beta during annealing or cycling')
    parser.add_argument('--lr_g', default=2e-4, type=float,
                        help='learning rate for all optimizers')
    parser.add_argument('--lr_d', default=2e-4, type=float,
                        help='learning rate for all optimizers')
    parser.add_argument('--lr_decay', default=0.95, type=float,
                        help='learning rate for all optimizers')
    parser.add_argument('--p_occlude', default=0.0, type=float,
                        help='number of joints to encode and decode')
    #gan
    parser.add_argument('--top_k', default=True, type=bool,
                        help='top k realistic samples to train generator')
    parser.add_argument('--top_k_gamma', default=0.99, type=float,
                        help='# decay rate of k')
    parser.add_argument('--top_k_min', default=0.5, type=float,
                        help='# least % of k')
    parser.add_argument('--disc_freq', default=5, type=int,
                        help='# disc steps for every gen step')
    parser.add_argument('--flip_labels_n_e', default=0, type=int,  
                        help='flip real fake labels for critic every n epochs')
    #schedulers
    parser.add_argument('--beta_warmup_epochs', default=10, type=int,
                        help='KLD weight warmup time. weight is 0 during this period')
    parser.add_argument('--beta_annealing_epochs', default=40, type=int,
                        help='KLD weight annealing time')
    parser.add_argument('--noise_level', default=0.0, type=float, 
                        help='percentage of noise to inject for critic training')
    # data files
    parser.add_argument('--train_file', default=f'{os.path.dirname(os.path.abspath(__file__))}/data/h36m_train_sh.h5', type=str,
                        help='abs path to training data file')
    parser.add_argument('--test_file', default=f'{os.path.dirname(os.path.abspath(__file__))}/data/h36m_test_sh.h5', type=str,
                        help='abs path to validation data file')
    # output
    # parser.add_argument('--save_dir', default=f'{os.path.dirname(os.path.abspath(__file__))}/../checkpoints', type=str,
    #                     help='path to save checkpoints')
    parser.add_argument('--log_image_interval', type=int, default=1,
                        help='log images during eval epoch falling in this interval')
    parser.add_argument('--validation_interval', type=int, default=5,
                        help='validate every nth epoch')
    # device
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='enable gpu if available')
    parser.add_argument('--pin_memory', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='pin memory to device')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='workers for data loader')
    parser.add_argument('--seed', default=400, type=int,
                        help='random seed')

    # fmt: on
    return parser


if __name__ == "__main__":
    main()
