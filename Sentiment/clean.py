from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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
	# porter = PorterStemmer()
	# stemmed = [porter.stem(word) for word in tokens]
	# return stemmed
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

def clean_dir(ogDir,newDir):
	for file in listdir(ogDir):
		try:
			text=load_doc(ogDir+file)
			tokenList=clean_doc(text)
			save_tokenlist(tokenList, newDir+file)
		except Exception as e:
			print(file,e)

#saved clean data 

if __name__ == "__main__":
	clean_dir("./og/train/pos/","./clean/train/pos/")
	clean_dir("./og/train/neg/","./clean/train/neg/")
	clean_dir("./og/test/pos/","./clean/test/pos/")
	clean_dir("./og/test/neg/","./clean/test/neg/")