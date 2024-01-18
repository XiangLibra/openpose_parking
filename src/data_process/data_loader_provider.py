import torch
from .coco import CocoDataSet


def create_data_loaders(opt):
    tr_dataset, te_dataset = create_data_sets(opt)
    train_loader = torch.utils.data.DataLoader(
        tr_dataset,
        batch_size=opt.batchSize,
        shuffle=True if opt.DEBUG == 0 else False,
        drop_last=True,
        num_workers=opt.nThreads
    )
    test_loader = torch.utils.data.DataLoader(
        te_dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=opt.nThreads
    )
    return train_loader, test_loader


def create_data_sets(opt):
    if opt.dataset == 'coco':
#        tr_dataset = CocoDataSet(opt.data, opt, 'train2017')
#        te_dataset = CocoDataSet(opt.data, opt, 'val2017')
        tr_dataset = CocoDataSet(opt.data, opt, opt.datatrain)
        te_dataset = CocoDataSet(opt.data, opt, opt.dataval)
    else:
        raise ValueError('Data set ' + opt.dataset + ' not available.')
    return tr_dataset, te_dataset
