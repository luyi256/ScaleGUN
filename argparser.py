import argparse


def argparser():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument("--dataset", default="cora", type=str)
    parser.add_argument("--path", default="./data/")
    parser.add_argument("--del_path_suffix", default="unlearning_data/")
    parser.add_argument("--analysis_path", default="./analysis")
    parser.add_argument("--trials", type=int, default=3)

    # args for preprocessing
    parser.add_argument("--axis_num", default=1, type=int, choices=[1, 0])

    # args for propagation
    parser.add_argument("--prop_algo", type=str,
                        choices=["power", "push", "MC"], default="push")
    parser.add_argument("--prop_step", default=3, type=int)
    parser.add_argument("--r", default=0.5, type=float)
    parser.add_argument("--decay", default=0.1, type=float)
    parser.add_argument("--RW", type=int, default=10000,
                        help="random walk times")
    parser.add_argument("--rmax", default=0.0, type=float)
    parser.add_argument("--ppr", default=False, action="store_true")
    parser.add_argument("--weight_mode", default="test",
                        type=str, choices=["decay", "avg", "test", "hetero"],)
    parser.add_argument("--num_threads", default=40, type=int)

    # args for model, loss function
    parser.add_argument("--layer", default=2, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--hidden", default=1024, type=int)
    # Set delta<0 here, then delta = 1/|E| will be set in the edge experiments, and delta = 1/|V| in the node experiments.
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1.0,
                        help="Eps coefficient for certified removal.")
    parser.add_argument("--lam", type=float, default=1e-2,
                        help="L2 regularization")
    parser.add_argument("--std", type=float, default=1e-1,
                        help="standard deviation for objective perturbation",)
    parser.add_argument("--noise_mode", type=str, default="data",
                        help="Data dependent noise or worst case noise [data/worst].",)

    # args for training
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--dropout", type=float,
                        default=0.3, help="dropout rate.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of optimization steps")
    parser.add_argument("--train_mode", type=str,
                        default="ovr", help="train mode [ovr/binary]")
    parser.add_argument("--Y_binary", type=str, default="0",
                        help="In binary mode, is Y_binary class or Y_binary_1 vs Y_binary_2 (i.e., 0+1).",)
    parser.add_argument("--init_method", type=str, default="kaiming",
                        help="init method for parameter w [kaiming/xavier/zero]",)
    parser.add_argument("--optimizer", type=str, default="LBFGS",
                        help="Choice of optimizer. [LBFGS/Adam]",)
    parser.add_argument("--wd", type=float, default=5e-4,
                        help="Weight decay factor for Adam")
    parser.add_argument("--dev", default=1, type=int,
                        help="cuda device, -1 for cpu")
    parser.add_argument("--patience", default=30, type=int)
    parser.add_argument("--train_batch", default=200000, type=int)
    parser.add_argument("--test_batch", default=100000, type=int)
    parser.add_argument("--checkid", default=0, type=int)

    # args for log
    parser.add_argument("--verbose", action="store_true",
                        default=False, help="verbosity in optimizer")
    parser.add_argument("--disp", type=int, default=10)
    parser.add_argument("--optuna", action="store_true", default=False,
                        help="Use optuna to optimize hyperparameters.",)

    # args for unlearing
    parser.add_argument("--compare_gnorm", action="store_true", default=False,
                        help="Compute norm of worst case and real gradient each round.",)
    parser.add_argument("--no_retrain", action="store_true", default=False)
    parser.add_argument("--num_batch_removes", default=5, type=int)
    parser.add_argument("--edge_idx_start", default=0, type=int)
    parser.add_argument("--damping", default=1e-4, type=float)
    parser.add_argument("--num_removes", default=1, type=int,
                        help="number of removed edges/nodes in each batch",)
    parser.add_argument("--removal_mode", default="node", type=str)
    parser.add_argument("--compare_retrain",
                        action="store_true", default=False)

    # args for attack
    parser.add_argument("--num_add", type=int, default=10000)
    parser.add_argument("--max_deg", type=int, default=5)
    parser.add_argument("--del_postfix", type=str, default="")
    args = parser.parse_args()

    if args.eps < 0:
        args.eps = 0.1/args.std
    return args
