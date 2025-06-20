"""
This type stub file was generated by pyright.
"""

from ultralytics.engine.validator import BaseValidator

class ClassificationValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a classification model.

    Notes:
        - Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Example:
        ```python
        from ultralytics.models.yolo.classify import ClassificationValidator

        args = dict(model="yolo11n-cls.pt", data="imagenet10")
        validator = ClassificationValidator(args=args)
        validator()
        ```
    """
    def __init__(self, dataloader=..., save_dir=..., pbar=..., args=..., _callbacks=...) -> None:
        """Initializes ClassificationValidator instance with args, dataloader, save_dir, and progress bar."""
        ...
    
    def get_desc(self): # -> LiteralString:
        """Returns a formatted string summarizing classification metrics."""
        ...
    
    def init_metrics(self, model): # -> None:
        """Initialize confusion matrix, class names, and top-1 and top-5 accuracy."""
        ...
    
    def preprocess(self, batch):
        """Preprocesses input batch and returns it."""
        ...
    
    def update_metrics(self, preds, batch): # -> None:
        """Updates running metrics with model predictions and batch targets."""
        ...
    
    def finalize_metrics(self, *args, **kwargs): # -> None:
        """Finalizes metrics of the model such as confusion_matrix and speed."""
        ...
    
    def postprocess(self, preds):
        """Preprocesses the classification predictions."""
        ...
    
    def get_stats(self): # -> dict[str, int | Any | float]:
        """Returns a dictionary of metrics obtained by processing targets and predictions."""
        ...
    
    def build_dataset(self, img_path): # -> ClassificationDataset:
        """Creates and returns a ClassificationDataset instance using given image path and preprocessing parameters."""
        ...
    
    def get_dataloader(self, dataset_path, batch_size): # -> InfiniteDataLoader:
        """Builds and returns a data loader for classification tasks with given parameters."""
        ...
    
    def print_results(self): # -> None:
        """Prints evaluation metrics for YOLO object detection model."""
        ...
    
    def plot_val_samples(self, batch, ni): # -> None:
        """Plot validation image samples."""
        ...
    
    def plot_predictions(self, batch, preds, ni): # -> None:
        """Plots predicted bounding boxes on input images and saves the result."""
        ...
    


