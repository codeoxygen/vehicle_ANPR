from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
from PIL import ImageFont
import re



# Paddleocr supports Chinese, English, French, German, Korean, and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # need to run only once to download and load the model into memory


def get_licsence_number(model , image_path):


    result = model.ocr(image_path, cls=True)
    print(result[0])
    if result[0] :

        for line in result:

            
            font = ImageFont.load_default()

            image = Image.open(image_path).convert('RGB')
            boxes = [detection[0] for line in result for detection in line]
            txts = [detection[1][0] for line in result for detection in line]
            scores = [detection[1][1] for line in result for detection in line]


            return result[0][0][-1][0]
    else:
        return None

            
        #im_show = draw_ocr(image, boxes, txts, scores ,font_path = '/Users/lahiru/Documents/Number plate detection model/trained_models/PaddleOCR/StyleText/fonts/ko_standard.ttf')
        #im_show = Image.fromarray(im_show)
        #im_show.save('result.jpg')


def pattern( number):
    provineces = ['NP' , 'SP' , 'EP' , 'WP', 'SG' , 'NC' , 'NW' , 'CP' , 'UP'  ]
    letters = number.split('-')[0].strip()
    if len(letters) > 2 and letters.isupper():
        province = letters[:len(letters) - 2]

        matches = [item for item in provineces if item.startswith(province)]
        if len(matches) > 0:
            p = matches[0]
            result = p + number[len(letters) - 2:] 
            return result
        else:
            return None




