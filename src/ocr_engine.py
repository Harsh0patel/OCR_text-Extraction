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

        self.ocr_vertical = PaddleOCR(
        use_angle_cls = True,
        lang='en',
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        text_det_box_thresh=0,
        )

    def predict(self, img):
        h, w = img.shape[:2]
        if h > w:
            output = self.ocr_vertical.predict(img)
        else:
            output = self.ocr.predict(img)
        return output