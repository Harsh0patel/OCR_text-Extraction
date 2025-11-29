from paddleocr import PaddleOCR

class PaddleOCREngine:
    def __init__(self):
        self.ocr = PaddleOCR(
            lang = 'en',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_box_thresh=0.9,
        )

