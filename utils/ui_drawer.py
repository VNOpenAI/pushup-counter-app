import cv2

ft = cv2.freetype.createFreeType2()
ft.loadFontData(fontFileName='Ubuntu-R.ttf',
                id=0)
ft.putText(img=img,
           text='Quick Fox',
           org=(15, 70),
           fontHeight=60,
           color=(255,  255, 255),
           thickness=-1,
           line_type=cv2.LINE_AA,
           bottomLeftOrigin=True)
