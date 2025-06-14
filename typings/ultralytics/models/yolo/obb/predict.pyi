"""
This type stub file was generated by pyright.
"""

from ultralytics.models.yolo.detect.predict import DetectionPredictor

class OBBPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.obb import OBBPredictor

        args = dict(model="yolo11n-obb.pt", source=ASSETS)
        predictor = OBBPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """
    def __init__(self, cfg=..., overrides=..., _callbacks=...) -> None:
        """Initializes OBBPredictor with optional model and data configuration overrides."""
        ...
    
    def construct_result(self, pred, img, orig_img, img_path): # -> Results:
        """
        Constructs the result object from the prediction.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and rotation angles.
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.

        Returns:
            (Results): The result object containing the original image, image path, class names, and oriented bounding boxes.
        """
        ...
    


