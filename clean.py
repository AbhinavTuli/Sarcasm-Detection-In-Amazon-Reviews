from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

#save all tokens from a doc into file
def save_tokenlist(tokenList, filename):
	data = '\n'.join(tokenList)
	# open file
	file = open(filename, 'w')
	# write text
	file.write(data)
	# close file
	file.close()


#saved clean data 
for file in listdir("./txt_sentoken/train/pos/"):
	try:
		text=load_doc("./txt_sentoken/train/pos/"+file)
		tokenList=clean_doc(text)
		save_tokenlist(tokenList, './clean/train/pos/'+file)
	except Exception as e:
		print(file)
		print(e)

for file in listdir("./txt_sentoken/train/neg/"):
	try:
		text=load_doc("./txt_sentoken/train/neg/"+file)
		tokenList=clean_doc(text)
		save_tokenlist(tokenList, './clean/train/neg/'+file)
	except Exception as e:
		print(file)
		print(e)

for file in listdir("./txt_sentoken/test/pos/"):
	try:
		text=load_doc("./txt_sentoken/test/pos/"+file)
		tokenList=clean_doc(text)
		save_tokenlist(tokenList, './clean/test/pos/'+file)
	except Exception as e:
		print(file)
		print(e)

for file in listdir("./txt_sentoken/test/neg/"):
	try:
		text=load_doc("./txt_sentoken/test/neg/"+file)
		tokenList=clean_doc(text)
		save_tokenlist(tokenList, './clean/test/neg/'+file)
	except Exception as e:
		print(file)
		print(e)