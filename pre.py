import re
import glob
from subprocess import Popen, PIPE, STDOUT
from os import walk
import os
import preprocess as pp

def generate_training_data(text):
	maxlen = 50
	sentences = []
	next_chars = []
	for i in range(0, len(text) - maxlen - 1):
	    sentences.append(text[i: i + maxlen])
	    next_chars.append(text[i+1 : i + maxlen])
	    sentences[i] = re.sub(r'[\n\t]',' ', sentences[i])
	    next_chars[i] = re.sub(r'[\n\t]',' ', next_chars[i])
	    print(sentences[i] + "\t" + next_chars[i])

path = 'testsuite'
files = []
valid_count = 0
for root, d_names, f_names in os.walk(path):
	for f in f_names:
		files.append(os.path.join(root, f))
for file in files:
	print ('--------------------------------------------------')
	print (file)
	if ('nocomment' in file or 'nospace' in file or 'nomacro' in file or 'raw' in file):
		command = 'rm ' + file
		p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
	text = open(file, 'r').read()
	text = remove_comment(text)
	text = replace_macro(text, file)
	text = remove_space(text)
	is_valid = verify_correctness(text, file, 'nospace')
	if (is_valid):
		valid_count += 1
		generate_training_data(text)	
print(valid_count)
