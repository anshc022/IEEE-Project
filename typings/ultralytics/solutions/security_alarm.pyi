"""
This type stub file was generated by pyright.
"""

from ultralytics.solutions.solutions import BaseSolution

class SecurityAlarm(BaseSolution):
    """
    A class to manage security alarm functionalities for real-time monitoring.

    This class extends the BaseSolution class and provides features to monitor
    objects in a frame, send email notifications when specific thresholds are
    exceeded for total detections, and annotate the output frame for visualization.

    Attributes:
       email_sent (bool): Flag to track if an email has already been sent for the current event.
       records (int): Threshold for the number of detected objects to trigger an alert.

    Methods:
       authenticate: Sets up email server authentication for sending alerts.
       send_email: Sends an email notification with details and an image attachment.
       monitor: Monitors the frame, processes detections, and triggers alerts if thresholds are crossed.

    Examples:
        >>> security = SecurityAlarm()
        >>> security.authenticate("abc@gmail.com", "1111222233334444", "xyz@gmail.com")
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = security.monitor(frame)
    """
    def __init__(self, **kwargs) -> None:
        """Initializes the SecurityAlarm class with parameters for real-time object monitoring."""
        ...
    
    def authenticate(self, from_email, password, to_email): # -> None:
        """
        Authenticates the email server for sending alert notifications.

        Args:
            from_email (str): Sender's email address.
            password (str): Password for the sender's email account.
            to_email (str): Recipient's email address.

        This method initializes a secure connection with the SMTP server
        and logs in using the provided credentials.

        Examples:
            >>> alarm = SecurityAlarm()
            >>> alarm.authenticate("sender@example.com", "password123", "recipient@example.com")
        """
        ...
    
    def send_email(self, im0, records=...): # -> None:
        """
        Sends an email notification with an image attachment indicating the number of objects detected.

        Args:
            im0 (numpy.ndarray): The input image or frame to be attached to the email.
            records (int): The number of detected objects to be included in the email message.

        This method encodes the input image, composes the email message with
        details about the detection, and sends it to the specified recipient.

        Examples:
            >>> alarm = SecurityAlarm()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> alarm.send_email(frame, records=10)
        """
        ...
    
    def monitor(self, im0):
        """
        Monitors the frame, processes object detections, and triggers alerts if thresholds are exceeded.

        Args:
            im0 (numpy.ndarray): The input image or frame to be processed and annotated.

        This method processes the input frame, extracts detections, annotates the frame
        with bounding boxes, and sends an email notification if the number of detected objects
        surpasses the specified threshold and an alert has not already been sent.

        Returns:
            (numpy.ndarray): The processed frame with annotations.

        Examples:
            >>> alarm = SecurityAlarm()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> processed_frame = alarm.monitor(frame)
        """
        ...
    


