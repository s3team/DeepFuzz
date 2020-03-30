import re
import glob
from subprocess import Popen, PIPE, STDOUT
from os import walk
import os
import glob
import preprocess as pp
import pdb

def generate_training_data(text):
	maxlen = 50
	sentences = []
	next_chars = []
	for i in range(0, len(text) - maxlen - 1):
	    sentences.append(text[i: i + maxlen])
	    next_chars.append(text[i + maxlen])
	    sentences[i] = re.sub(r'[\n\t]',' ', sentences[i])
	    next_chars[i] = re.sub(r'[\n\t]',' ', next_chars[i])
	    print(sentences[i] + "\t" + next_chars[i])

path = './gcc/gcc/testsuite'
files = []
valid_count = 0
for root, d_names, f_names in os.walk(path):
	for f in f_names:
		files.append(os.path.join(root, f))
for file in files:
	if ('nocomment' in file or 'nospace' in file or 'nomacro' in file or 'raw' in file):
		command = 'rm ' + file
		p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
	if (not file.endswith('.c') and not file.endswith('.h') and not file.endswith('.C')):
		continue
	text = open(file, 'r', encoding='iso-8859-1').read()
	text = pp.remove_comment(text)
	text = pp.replace_macro(text, file)
	text = pp.remove_space(text)
	is_valid = pp.verify_correctness(text, file, 'nospace')
	if (is_valid):
		valid_count += 1
		generate_training_data(text)	
print(valid_count)
