from PIL import Image, ImageEnhance

def ig_filter(s):
    image = Image.open(s)
    img=image.copy()
    #Adjust brightness
    enhancer_brightness = ImageEnhance.Brightness(img)
    img = enhancer_brightness.enhance(0.5)

    #Adjust contrast
    enhancer_contrast = ImageEnhance.Contrast(img)
    img = enhancer_contrast.enhance(1.5)

    #Adjust saturation
    enhancer_saturation = ImageEnhance.Color(img)
    img = enhancer_saturation.enhance(1.5)

    image.show()
    img.show()

s = 'switzerland.jpg'
ig_filter(s)
