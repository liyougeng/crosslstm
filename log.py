import os

def Log(modelfile, text):
	if modelfile[-6] == '_': end_pos = -6
	else: end_pos = -7
	logfile = "" + modelfile[13:end_pos]
	logfile = "./logs/" + logfile + '.log'
	with open(logfile, 'a') as fd:
		fd.write("%s\n" % text)
	print(text)
