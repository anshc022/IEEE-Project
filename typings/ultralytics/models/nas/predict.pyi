"""
This type stub file was generated by pyright.
"""

from ultralytics.engine.predictor import BasePredictor

class NASPredictor(BasePredictor):
    """
    Ultralytics YOLO NAS Predictor for object detection.

    This class extends the `BasePredictor` from Ultralytics engine and is responsible for post-processing the
    raw predictions generated by the YOLO NAS models. It applies operations like non-maximum suppression and
    scaling the bounding boxes to fit the original image dimensions.

    Attributes:
        args (Namespace): Namespace containing various configurations for post-processing.

    Example:
        ```python
        from ultralytics import NAS

        model = NAS("yolo_nas_s")
        predictor = model.predictor
        # Assumes that raw_preds, img, orig_imgs are available
        results = predictor.postprocess(raw_preds, img, orig_imgs)
        ```

    Note:
        Typically, this class is not instantiated directly. It is used internally within the `NAS` class.
    """
    def postprocess(self, preds_in, img, orig_imgs): # -> list[Any]:
        """Postprocess predictions and returns a list of Results objects."""
        ...
    


