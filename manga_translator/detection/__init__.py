import numpy as np

from .default import DefaultDetector
from .dbnet_convnext import DBConvNextDetector
from .ctd import ComicTextDetector
from .craft import CRAFTDetector
from .paddle import PaddleDetector
from .none import NoneDetector
from .gemini import GeminiDetector # Import the new detector
from .common import CommonDetector, OfflineDetector
from ..config import Detector, DetectorConfig # Import DetectorConfig

DETECTORS = {
    Detector.default: DefaultDetector,
    Detector.dbconvnext: DBConvNextDetector,
    Detector.ctd: ComicTextDetector,
    Detector.craft: CRAFTDetector,
    Detector.paddle: PaddleDetector,
    Detector.gemini: GeminiDetector, # Add Gemini to the map
    Detector.none: NoneDetector,
}
detector_cache = {}

def get_detector(key: Detector, *args, **kwargs) -> CommonDetector:
    if key not in DETECTORS:
        raise ValueError(f'Could not find detector for: "{key}". Choose from the following: %s' % ','.join(DETECTORS))
    if not detector_cache.get(key):
        detector = DETECTORS[key]
        detector_cache[key] = detector(*args, **kwargs)
    return detector_cache[key]

async def prepare(detector_key: Detector):
    detector = get_detector(detector_key)
    if isinstance(detector, OfflineDetector):
        await detector.download()

async def dispatch(detector_key: Detector, image: np.ndarray, config: DetectorConfig, device: str = 'cpu', verbose: bool = False):
    """
    Dispatches the detection task to the appropriate detector implementation.

    Args:
        detector_key: The enum key for the desired detector.
        image: Input image as a NumPy array.
        config: The DetectorConfig object containing all detection parameters.
        device: The device to run on ('cpu', 'cuda', 'mps').
        verbose: Enable verbose logging.

    Returns:
        The result from the detector's detect method.
    """
    detector = get_detector(detector_key)

    # Load offline models if necessary
    if isinstance(detector, OfflineDetector):
        # Specific loading logic for PaddleDetector if needed, otherwise generic load
        if isinstance(detector, PaddleDetector):
             # PaddleDetector load might need specific params from config if its load method requires them
             # Assuming load only needs device for now, adjust if PaddleDetector.load changes
             await detector.load(device, text_threshold=config.text_threshold, box_threshold=config.box_threshold, unclip_ratio=config.unclip_ratio, invert=config.det_invert, verbose=verbose)
        else:
             await detector.load(device) # Generic load for other offline detectors

    # Call the detect method
    # For Gemini, pass the whole config object as its detect method expects it
    if detector_key == Detector.gemini:
        # Ensure the GeminiDetector's detect method signature matches this call
        return await detector.detect(image, detect_config=config, verbose=verbose)
    else:
        # For other detectors, extract parameters from the config object
        return await detector.detect(
            image,
            detect_size=config.detection_size,
            text_threshold=config.text_threshold,
            box_threshold=config.box_threshold,
            unclip_ratio=config.unclip_ratio,
            invert=config.det_invert,
            gamma_correct=config.det_gamma_correct,
            rotate=config.det_rotate,
            auto_rotate=config.det_auto_rotate,
            verbose=verbose
        )

async def unload(detector_key: Detector):
    detector_cache.pop(detector_key, None)
