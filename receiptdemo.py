#!/usr/bin/env python
#
# Copyright 2009 Facebook
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import logging
import tornado.escape
import tornado.ioloop
import tornado.web
import cv2
import numpy as np
from PIL import Image
import io
from PIL import Image
import os
import sys
import pytesseract
import re
import dateutil.parser as dparser
from difflib import get_close_matches
from money_parser import price_str

from tornado.options import define, options, parse_command_line

define("port", default=8888, help="run on the given port", type=int)
define("debug", default=False, help="run in debug mode")

lang = "eng"

pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
config = "--psm 3 --oem 2 --user-words user-words.txt --user-patterns user-patterns --tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz:$.-0123456789"

price_regex = re.compile("[\s\$][\d\s*]+\.[\d\s*]{2}")
time_regex = re.compile("[0-9]{1,2}:[0-9]{2}\s*(AM|PM)")
digits_regex = re.compile("\d+")

stores = []
with open('stores.txt') as f:
    stores = f.readlines()
stores = [store.strip() for store in stores]

items_dict = []
with open('items.txt') as f:
    items_dict = f.readlines()
items_dict = [item.strip() for item in items_dict]

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

    def post(self):
        image_file = self.request.files['image'][0]
        original_fname = image_file['filename']

        nparr = np.fromstring(image_file['body'], np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # contours = recognize_text(image)
        # cv2.imshow('contours',contours)
        # cv2.waitKey(0)
        image = Image.fromarray(image)
        text = pytesseract.image_to_string(image, lang=lang, config=config)
        print(text)

        prices = []
        items = []
        date = None
        time = None
        lines = text.splitlines()
        store = find_store(lines)
        if store != None:
            print(store)

        for line in text.splitlines():
            line = line.strip()
            if (len(line) > 0):
                match = price_regex.search(line)
                if match != None:
                    price = match.group().replace(" ", "")
                    index = match.span()[0]
                    product = line[:index]
                    product = re.sub(r'[^0-9a-zA-Z/ ]+', '', product).strip()
                    findItem = find_item(product)
                    if findItem != None:
                        product = findItem
                    prices.append(price_str(price))
                    items.append(product)
                else:
                    try:
                        if time == None:
                            match = time_regex.search(line)
                            if match != None:
                                time = match.group()
                                print(time)
                        matchItem = find_item(line)
                        if matchItem != None:
                            quantity = re.search(digits_regex, line[:4])
                            if quantity != None:
                                matchItem = quantity.group() + ' ' + matchItem
                                print(matchItem)
                            prices.append(float(0))
                            items.append(matchItem)
                        if date == None:
                            date = dparser.parse(line,fuzzy=True, dayfirst=True)
                            print(date)
                    except:
                        pass
        self.render("result.html", prices=prices, items=items, date=date, time=time, store=store)

def find_item(line, accuracy=0.6):
    """
    :param keyword: str
        The keyword string to look for
    :param accuracy: float
        Required accuracy for a match of a string with the keyword
    :return: str
        Returns the first line in lines that contains a keyword.
        It runs a fuzzy match if 0 < accuracy < 1.0
    """
    matches = get_close_matches(line.lower(), items_dict, 1, accuracy)
    if matches:
        return matches[0]
    return None

def find_store(lines, accuracy=0.6):
    """
    :param keyword: str
        The keyword string to look for
    :param accuracy: float
        Required accuracy for a match of a string with the keyword
    :return: str
        Returns the first line in lines that contains a keyword.
        It runs a fuzzy match if 0 < accuracy < 1.0
    """

    for line in lines:
        matches = get_close_matches(line.lower(), stores, 1, accuracy)
        if matches:
            return matches[0]
    return None

def recognize_text(original):
    idcard = original
    gray = cv2.cvtColor(idcard, cv2.COLOR_BGR2GRAY)

    # Morphological gradient:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    # Binarization
    ret, binarization = cv2.threshold(opening, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Connected horizontally oriented regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(binarization, cv2.MORPH_CLOSE, kernel)

    # find countours
    image, contours, hierarchy = cv2.findContours(
        connected, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    return image

def main():
    parse_command_line()
    app = tornado.web.Application(
        [
            (r"/", MainHandler),
        ],
        cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
        template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        xsrf_cookies=True,
        debug=options.debug,
    )
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
