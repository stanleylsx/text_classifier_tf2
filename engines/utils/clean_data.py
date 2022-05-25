# -*- coding: utf-8 -*- 
# @Time : 2021/3/17 16:37  
# @Author : Stanley  
# @EMail : gzlishouxian@gmail.com
# @File : clean_data.py
# @Software: PyCharm
import re


def filter_word(raw_word):
    if raw_word in ['\t', '']:
        return False
    if not re.search(r'^[\u4e00-\u9fa5_a-zA-Z\d]+$', raw_word):
        return False
    else:
        return True


def filter_char(char, remove_sp=True):
    if char in ['\t', '']:
        return False
    if remove_sp:
        if re.search(r'[\u4e00-\u9fa5_a-zA-Z\d]', char):
            return True
        else:
            return False
    else:
        return False
