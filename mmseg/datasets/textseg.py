from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class TextSegDataset(CustomDataset):

    CLASSES = ('Background', 'Text')


    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(TextSegDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_maskfg.png',
            reduce_zero_label=False,
            **kwargs)
