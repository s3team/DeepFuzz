import numpy as np
import os
import re
from subprocess import Popen, PIPE, STDOUT

def remove_comment(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

def verify_correctness(text, filename, mode):
    name = filename[:-2]
    postfix = filename[-2:]
    filename = name + '_' + mode + postfix
    f = open(filename, 'w')
    command = 'gcc -c -w ' + filename
    f.write(text)
    p1 = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    stderr = str(p1.stdout.read())
    print(stderr)
    if ('internal compiler error' in stderr):
        return True
    if ('error' in stderr):
        return False
    else:
        return True

def replace_macro(text, filename):
    headers = re.findall(r'#include .*\n', text)
    text = re.sub(r'#include .*\n', '', text)
    postfix = filename[-2:]
    tempfile = 'temp' + postfix
    print(text, file=open(tempfile, 'w'))
    command = 'gcc -E ' + tempfile + ' > temp'
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    text = open('temp', 'r').read()
    text = re.sub(r'# \d+ .*\n', '', text)
    for header in headers:
        text = header + text
    return text

def remove_space(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s', ' ', text)
    text = re.sub(r'\\t', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text
