import argparse

from dataset.nyuv2_raw.nyuv2_raw_datamodule import NYUv2RawDataModule
import configargparse


def add_train_args(parent_parser):
    parser = parent_parser.add_argument_group("Training")
    parser.add_argument(
        "--adam_eps", type=float, help="epsilon in Adam optimizer", default=1e-6
    )
    parser.add_argument("--batch_size", type=int, help="batch size", default=4)
    parser.add_argument("--num_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument(
        "--learning_rate", type=float, help="initial learning rate", default=1e-4
    )
    return parent_parser


def main():
    argparser = configargparse.ArgParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add(
        "-c",
        "--config-file",
        is_config_file=True,
        help="config file path",
        default="train-config.yml",
    )
    argparser = add_train_args(argparser)
    argparser = NYUv2RawDataModule.add_argparse_args(argparser)

    args = argparser.parse_args()

    dm = NYUv2RawDataModule(args)
    dm.setup()
    for mode in ["train", "val"]:
        dm.check_dataloader(mode)


if __name__ == "__main__":
    main()
