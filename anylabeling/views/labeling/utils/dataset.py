def open_dataset_split(self):
    from anylabeling.views.labeling.widgets.dataset_split_dialog import (
        DatasetSplitDialog,
    )

    dialog = DatasetSplitDialog(self)
    dialog.exec()
