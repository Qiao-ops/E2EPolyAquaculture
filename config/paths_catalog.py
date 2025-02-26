import os
import os.path as osp

class DatasetCatalog(object):

    DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__),
                '..','..','data'))
    
    DATASETS = {
        'crowdai_train_small': {
            'img_dir': 'crowdai/train/images',
            'ann_file': 'crowdai/train/annotation-small.json'
        },
        'crowdai_test_small': {
            'img_dir': 'crowdai/val/images',
            'ann_file': 'crowdai/val/annotation-small.json'
        },
        'crowdai_train': {
            'img_dir': 'crowdai/train/images',
            'ann_file': 'crowdai/train/annotation.json'
        },
        'crowdai_test': {
            'img_dir': 'crowdai/val/images',
            'ann_file': 'crowdai/val/annotation.json'
        },
        'inria_train': {
            'img_dir': 'inria/train/images',
            'ann_file': 'inria/train/annotation.json',
        },
        'inria_test': {
            'img_dir': 'coco-Aerial/val/images',
            'ann_file': 'coco-Aerial/val/annotation.json',
        },
        'lyg_train': {
            # 'img_dir': 'lyg/cut_512/train/images',
            # 'ann_file': 'lyg/cut_512/train/annotation.json',
            'img_dir': 'lyg/cut_2048/train_512/images',
            'ann_file': 'lyg/cut_2048/train_512/annotation.json',
        },
        'lyg_test': {
            'img_dir': 'lyg/cut_512/val/images',
            'ann_file': 'lyg/cut_512/val/annotation.json',
        },
        'lyg_test_eval': {
            # 'img_dir': 'lyg/cut_2048/val/images',
            # 'ann_file': 'lyg/cut_2048/val/annotation.json',
            'img_dir': 'lyg/cut_2048/train_512/images',
            'ann_file': 'lyg/cut_2048/train_512/annotation.json',
        }
    }

    @staticmethod
    def get(name):
        assert name in DatasetCatalog.DATASETS
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]

        args = dict(
            root = osp.join(data_dir,attrs['img_dir']),
            ann_file = osp.join(data_dir,attrs['ann_file'])
        )

        if 'train' in name:
            return dict(factory="TrainDataset",args=args)
        if 'test' in name and 'ann_file' in attrs:
            return dict(factory="TestDatasetWithAnnotations",
                        args=args)
        raise NotImplementedError()
