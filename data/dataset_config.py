from mri_dataset import MedLMDBDataset, list_file_tree


def get_train_class_dataset():
    train_dataset = MedLMDBDataset(
        lmdb_root='/home/khtao/Foundation_Model_cache/all_datasets_128',
        dir_root='/home/khtao/mri_datasets',
        meta_path=list_file_tree('/mnt/WorkCenter/PycharmProjects/2025-11/MetaMAE/report_with_multilabel', 'xlsx'),
        return_key='class',
        contained_nan=True,
        strict_center=False,
        mode='train',

    )
    return train_dataset


def get_val_class_dataset():
    train_dataset = MedLMDBDataset(
        lmdb_root='/home/khtao/Foundation_Model_cache/all_datasets_128',
        dir_root='/home/khtao/mri_datasets',
        meta_path=[
            '/mnt/WorkCenter/PycharmProjects/2025-11/MetaMAE/report_with_multilabel/EC_shanwei_withtxt_adjust.xlsx',
            '/mnt/WorkCenter/PycharmProjects/2025-11/MetaMAE/report_with_multilabel/风险分层-深汕妇科_ori.xlsx',
            '/mnt/WorkCenter/PycharmProjects/2025-11/MetaMAE/report_with_multilabel/CC_guangzhoushizhongliuyiyuan_withtxt_adjust.xlsx',
            '/mnt/WorkCenter/PycharmProjects/2025-11/MetaMAE/report_with_multilabel/CC_shanwei_withtxt_adjust_buchongbaogao20251201.xlsx',
            '/mnt/WorkCenter/PycharmProjects/2025-11/MetaMAE/report_with_multilabel/zigongjiliu_shanwei_withtxt_adjust.xlsx',
            '/mnt/WorkCenter/PycharmProjects/2025-11/MetaMAE/report_with_multilabel/深汕_盆腔_MRI_整理后数据_整理Kang.xlsx',
        ],
        return_key='class',
        contained_nan=True,
        strict_center=True,
        mode='val',
    )
    return train_dataset


def get_train_text_dataset():
    train_dataset = MedLMDBDataset(
        lmdb_root='/home/khtao/Foundation_Model_cache/all_datasets_128',
        dir_root='/home/khtao/mri_datasets',
        meta_path=list_file_tree('/mnt/WorkCenter/PycharmProjects/2025-11/MetaMAE/report_with_multilabel', 'xlsx'),
        return_key='structure',
        contained_nan=True,
        strict_center=False,
        mode='train',

    )
    return train_dataset


def get_val_text_dataset():
    train_dataset = MedLMDBDataset(
        lmdb_root='/home/khtao/Foundation_Model_cache/all_datasets_128',
        dir_root='/home/khtao/mri_datasets',
        meta_path=[
            '/mnt/WorkCenter/PycharmProjects/2025-11/MetaMAE/report_with_multilabel/EC_shanwei_withtxt_adjust.xlsx',
            '/mnt/WorkCenter/PycharmProjects/2025-11/MetaMAE/report_with_multilabel/风险分层-深汕妇科_ori.xlsx',
            '/mnt/WorkCenter/PycharmProjects/2025-11/MetaMAE/report_with_multilabel/CC_guangzhoushizhongliuyiyuan_withtxt_adjust.xlsx',
            '/mnt/WorkCenter/PycharmProjects/2025-11/MetaMAE/report_with_multilabel/CC_shanwei_withtxt_adjust_buchongbaogao20251201.xlsx',
            '/mnt/WorkCenter/PycharmProjects/2025-11/MetaMAE/report_with_multilabel/zigongjiliu_shanwei_withtxt_adjust.xlsx',
            '/mnt/WorkCenter/PycharmProjects/2025-11/MetaMAE/report_with_multilabel/深汕_盆腔_MRI_整理后数据_整理Kang.xlsx',
        ],
        return_key='structure',
        contained_nan=True,
        strict_center=True,
        mode='val',
    )
    return train_dataset


if __name__ == '__main__':
    import torch

    train_dataset = get_train_class_dataset()
    task_name = '预训练'
    torch.save({
        'task_name': task_name,
        'meta_data': train_dataset.meta_data,
        'meta_path': train_dataset.meta_path,
        'return_key': train_dataset.return_key, },
        task_name + '_train_dataset.pt'
    )
