from paddleocr import PaddleOCR
model = PaddleOCR(lang = 'en',use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)


def predict(img):
    output = model.predict(img)
    return output