import os
import numpy as np
import cv2
import logging
import json
from PIL import Image
from io import BytesIO
from typing import List, Tuple

# Attempt to import google.generativeai, handle import error if library not installed
try:
    import google.generativeai as genai
except ImportError:
    genai = None # Set to None if import fails

from .common import CommonDetector
from ..utils import TextBlock, log_entry_exit, logger
from ..config import DetectorConfig # Assuming gemini_model is added here

# Configure logging for this module
logger = logging.getLogger(__name__)

class GeminiDetector(CommonDetector):
    """
    Detector using the Google Gemini Vision API for combined detection and OCR.
    Requires the GEMINI_API_KEY environment variable to be set.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if genai is None:
            raise ImportError("The 'google-generativeai' library is required to use the Gemini detector. Please install it.")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. This is required for the Gemini detector.")

        try:
            genai.configure(api_key=api_key)
            # Test connection or model availability if possible (optional)
            # Example: list models to ensure API key is valid
            # genai.list_models()
            logger.info("Gemini API configured successfully.")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            raise ConnectionError(f"Failed to configure Gemini API: {e}")

    @log_entry_exit(logger)
    async def detect(self, image: np.ndarray, detect_config: DetectorConfig, verbose: bool = False) -> Tuple[List[TextBlock], np.ndarray, np.ndarray]:
        """
        Detects text regions and performs OCR using the Gemini Vision API.

        Args:
            image: Input image in NumPy array format (RGB).
            detect_config: Detector configuration object containing gemini_model name.
            verbose: Enable verbose logging.

        Returns:
            A tuple containing:
            - List of TextBlock objects with coordinates and recognized text.
            - Raw mask (generated from bounding boxes).
            - Refined mask (same as raw mask in this implementation).
        """
        textlines: List[TextBlock] = []
        height, width = image.shape[:2]
        mask_raw = np.zeros((height, width), dtype=np.uint8)

        try:
            model_name = detect_config.gemini_model
            model = genai.GenerativeModel(model_name)

            # Convert NumPy array (RGB) to PIL Image bytes
            pil_image = Image.fromarray(image)
            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format='PNG') # Use PNG or JPEG
            img_bytes = img_byte_arr.getvalue()

            # Prepare the prompt for Gemini
            # Requesting JSON output for easier parsing
            prompt = """Analyze this manga image. Identify all distinct text bubbles or text regions. For each region, provide:
1.  A precise bounding polygon (list of [x, y] coordinates).
2.  The recognized text content within that polygon.
Output the result as a JSON list, where each item represents a text region and has 'polygon' and 'text' keys. Example: [{"polygon": [[x1, y1], [x2, y2], ...], "text": "some text"}, ...]"""

            # Make the API call
            logger.info(f"Calling Gemini API model: {model_name}")
            response = await model.generate_content_async([prompt, {'mime_type': 'image/png', 'data': img_bytes}]) # Use await for async call

            # --- Response Parsing ---
            if not response.parts:
                 logger.warning("Gemini API returned no parts in the response.")
                 return textlines, mask_raw, mask_raw # Return empty results

            # Clean potential markdown/JSON markers
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            logger.debug(f"Gemini Raw Response Text:\n{response_text}")

            try:
                detected_regions = json.loads(response_text)
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse JSON response from Gemini: {json_err}")
                logger.error(f"Problematic response text: {response_text}")
                # Optionally, try regex or other methods to salvage partial data if needed
                return textlines, mask_raw, mask_raw # Return empty on parse failure

            if not isinstance(detected_regions, list):
                 logger.warning(f"Gemini response was not a JSON list as expected: {type(detected_regions)}")
                 return textlines, mask_raw, mask_raw

            # Process parsed regions
            for region_data in detected_regions:
                if not isinstance(region_data, dict) or 'polygon' not in region_data or 'text' not in region_data:
                    logger.warning(f"Skipping invalid region data from Gemini: {region_data}")
                    continue

                polygon_pts = region_data.get('polygon')
                text_content = region_data.get('text', '').strip()

                if not isinstance(polygon_pts, list) or not text_content:
                    logger.warning(f"Skipping region with invalid polygon or empty text: {region_data}")
                    continue

                try:
                    # Convert polygon points to NumPy array of int32
                    pts = np.array(polygon_pts, dtype=np.int32)
                    if pts.ndim != 2 or pts.shape[1] != 2:
                         logger.warning(f"Skipping region with invalid polygon shape: {pts.shape}")
                         continue

                    # Create TextBlock
                    text_block = TextBlock(pts=pts, text=text_content)
                    textlines.append(text_block)

                    # Draw polygon on mask
                    cv2.fillPoly(mask_raw, [pts], 255)

                except Exception as e:
                    logger.warning(f"Error processing region data {region_data}: {e}")
                    continue

            logger.info(f"Gemini detected {len(textlines)} text regions.")

        except genai.types.generation_types.StopCandidateException as stop_ex:
             logger.warning(f"Gemini API call stopped, potentially due to safety settings or prompt issues: {stop_ex}")
             # Return empty results as the generation was blocked/incomplete
             return [], np.zeros((height, width), dtype=np.uint8), np.zeros((height, width), dtype=np.uint8)
        except Exception as e:
            logger.error(f"Error during Gemini API call or processing: {e}", exc_info=True)
            # Depending on ignore_errors flag (not directly accessible here, maybe pass via config?),
            # either raise or return empty results. Returning empty for now.
            return [], np.zeros((height, width), dtype=np.uint8), np.zeros((height, width), dtype=np.uint8)

        # For Gemini, raw mask and refined mask are the same initially
        return textlines, mask_raw, mask_raw.copy()

    async def download(self):
        # No model download needed for API-based detector
        pass

    async def load(self, device: str = 'cpu', *args, **kwargs):
        # No model loading needed for API-based detector
        pass

    async def unload(self):
        # No model unloading needed
        pass
