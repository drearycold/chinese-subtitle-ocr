import logging
import os
import sys

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import cv2
import numpy as np
import yaml
import pytesseract
import json


from detection import Detection
from recognition import Recognition

FONTS = ["NotoSansCJK-Regular.ttc", "wqy-zenhei.ttc", "SourceHanSansCN-Regular.otf", "simsun.ttc"]
FONTS = ["SourceHanSansCN-Regular.otf"]
FONT_COLOR = "white"
RECTANGLE_COLOR = "green"


def load_font(font_name, size):
    try:
        font = ImageFont.truetype(font_name, size)
    except IOError:
        logging.warning("Font {} not found".format(font_name))
        return None

    return font


def draw_text(draw, font, font2, pos_x, pos_y, text, prob, cyk=True):
    start, stop = pos_x
    y_start, y_end = pos_y
    draw.rectangle([(start, y_start), (stop, y_end)], outline=RECTANGLE_COLOR)
    draw.rectangle([(start + 1, y_start + 1), (stop - 1, y_end - 1)], outline=RECTANGLE_COLOR)
    draw.rectangle([(start + 2, y_start + 2), (stop - 2, y_end - 2)], outline=RECTANGLE_COLOR)
    probability = str(int(prob * 100))
    if cyk:
        draw.text((start, y_start - (stop - start)), text, fill=FONT_COLOR, font=font)
        draw.text((start, y_start - 1.5 * (stop - start)), probability + "%", fill=FONT_COLOR, font=font2)
    else:
        logging.warn("Detected character {} ({} %)".format(text, probability))


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    with open(sys.argv[1], "r") as config_file:
        cfg = yaml.safe_load(config_file)

    print(str(cfg))

    det_cfg = cfg["detection"]
    rec_cfg = cfg["recognition"]

    logging.basicConfig(format="%(asctime)s %(module)-12s %(levelname)-8s %(message)s")

    logging.warn("Starting detection")


    detection = Detection(det_cfg)

    found_frames = detection.detect_subtitle_region(cfg["video"])

    y_start, y_end = detection.get_subtitle_region()
    char_width = detection.get_char_width()
    char_dist = detection.get_char_dist()
    if char_width == 0 or char_dist == 0:
        logging.error("Char width is 0")
        return

    logging.warn(
        "Found y pos ({}, {}), character width {}, character distance {}".format(y_start, y_end, char_width, char_dist))

    recognition = Recognition(rec_cfg["model"], rec_cfg["weights"], rec_cfg["dictionary"])

    cyk = True
    for index, f in enumerate(FONTS):
        font = load_font(f, char_width)
        font2 = load_font(f, char_width // 2)
    if font is None:
        logging.error("No CYK font found")
        cyk = False
    else:
        logging.warn("Loaded font {}".format(FONTS[index]))

    cap = cv2.VideoCapture(cfg["video"])
    save_image_seq = cfg["video_offset_start"]
    save_image_seq_end = cfg["video_offset_end"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, save_image_seq)
    vout = cv2.VideoWriter(cfg["output_sub_video"], cv2.VideoWriter_fourcc(*'mp4v'), 29.97, (1920,1080-y_start+120))
    vout.set(cv2.VIDEOWRITER_PROP_QUALITY, 0.1)
    print(vout)

    custom_config = r'--psm 7 -l chi_sim'
    frames_ocr = {}
    #for frame in found_frames:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        text = []
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        x_start = 1920
        x_end = 0
        for char_region, start, stop in detection.detect_char_regions(frame[y_start:y_end, ], save_image=False, save_image_name="fill/seq_{}_{:06d}.tiff".format("{}", save_image_seq)):
            if x_start > start:
                x_start = start
            if x_end < stop:
                x_end = stop
            continue
            res = recognition.recognize_character(char_region)
            text.append((start, stop, res[1], res[2]))
            logging.warn("Detected Region {} {} in ({} {})".format(start, stop, y_start, y_end))

        save_image_seq += 1
        if save_image_seq > save_image_seq_end:
            break

        for start, stop, char, prob in text:
            draw.rectangle([(start, y_start), (stop, y_end)], outline=RECTANGLE_COLOR)
            draw.rectangle([(start + 1, y_start + 1), (stop - 1, y_end - 1)], outline=RECTANGLE_COLOR)
            draw.rectangle([(start + 2, y_start + 2), (stop - 2, y_end - 2)], outline=RECTANGLE_COLOR)

            probability = str(int(prob * 100)) + "%"
            if cyk:
                draw.text((start, y_start - (stop - start)), char, fill=FONT_COLOR, font=font)
                draw.text((start, y_start - 1.5 * (stop - start)), probability, fill=FONT_COLOR, font=font2)
            
            #logging.warn("Detected character {} ({})".format(char, probability))

        #cv2.imshow('image', np.array(img))
        #cv2.resizeWindow('image', int(1920/2), int(1080/2))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        vout.write(frame[y_start-120:1080, ])
        if x_start < x_end:
            gray = cv2.cvtColor(frame[y_start:y_end, x_start:x_end], cv2.COLOR_BGR2GRAY)
            #gray = img

            # threshhold
            ret,bin = cv2.threshold(gray,245,255,cv2.THRESH_BINARY)

            # closing
            kernel = np.ones((3,3),np.uint8)
            closing = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)

            # invert black/white
            inv = cv2.bitwise_not(closing)

            img_rgb = cv2.cvtColor(inv, cv2.COLOR_GRAY2RGB)
            #print(img_rgb)
            data_xml = pytesseract.image_to_alto_xml(img_rgb, config=custom_config)
            print(str(save_image_seq) + " " + data_xml.decode('utf-8'))
            #print(str(i) + " " + json.dumps(data_xml.decode('utf-8')))
            frames_ocr[save_image_seq] = data_xml.decode('utf-8')

    cap.release()
    vout.release()

    with open(cfg['output_sub_ocr'], 'w') as outfile:
        json.dump(frames_ocr, outfile, sort_keys=True, indent=2)

if __name__ == "__main__":
    main()
