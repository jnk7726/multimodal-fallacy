import logging
from pathlib import Path
from typing import List

from mamkit.data.datasets import SplitInfo, MMUSEDFallacy, InputMode


def loading_data_example():
    base_data_path = Path(__file__).parent.parent.resolve().joinpath('data')
    loader = MMUSEDFallacy(task_name='afd',
                           input_mode=InputMode.TEXT_ONLY,
                           base_data_path=base_data_path)
    logging.info(loader.data)


def custom_splits(
        self,
) -> List[SplitInfo]:
    train_df = self.data.iloc[:50]
    val_df = self.data.iloc[50:100]
    test_df = self.data.iloc[100:]
    fold_info = self.build_info_from_splits(train_df=train_df, val_df=val_df, test_df=test_df)
    return [fold_info]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    loading_data_example()
    # loading_predefined_splits()
    # defining_custom_splits_example()
