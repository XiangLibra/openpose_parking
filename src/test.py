import numpy as np
import torch
import random
from opts.base_opts import Opts
from data_process.data_loader_provider import create_data_loaders
from model.model_provider import create_model, create_optimizer
from evaluation.test_net import eval_net
from evaluation.coco import eval_COCO

def main():
    # Seed all sources of randomness to 0 for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)

    opt = Opts().parse()

    # Create data loaders
    # _, test_loader = create_data_loaders(opt)

    # Create nn
    model, _, _ = create_model(opt)
    model = model.cuda()
    TestImage="../data/val2017_test/"
    #TestImage="D:/jupyter note/part-affinity/data/val2017_test"

    # Get nn outputs
    outputs = eval_net( model, opt,TestImage)


if __name__ == '__main__':
    main()
