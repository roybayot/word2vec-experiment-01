
"""This is before I turned it into a script. Most of this file has been patterned from github.com/manasRK/glove-gensim"""


# In[12]:
import sys
import getopt


# In[13]:

def getNumLines(inputFile):
    with open(inputFile) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
    #num_lines = len([1 for line in open(inputFile)])
    #return num_lines


# In[21]:

def getNumDims(inputFile):
    a = open(inputFile)
    oneLine = a.readline()
    listOfTermsInLine = oneLine.split(" ")
    lenOneLine = len(listOfTermsInLine)
    dim = lenOneLine-1
    a.close()
    return dim


# In[ ]:

def writeTempLengthFile(numLines, numDims):
    temp_file = "temp_file_with_length.txt"
    with open(temp_file,'w') as f:
        f.write(str(numLines)+ " " +str(numDims) + '\n')
    f.close()
    print "Done writing temp file." 
    return temp_file


# In[ ]:

def writeConvertedFile(inputFile, tempFile, outputFile):
    filenames = [tempFile, inputFile]
    
    with open(outputFile, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    outfile.close()
    return 0


# In[22]:

def getRelevantFiles(argv):
   inputFile = ''
   outputFile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print './makeTwitterGloveFileCompatibleForGensim.py -i <input file> -o <output file>'
      print 'The input file is one of the models from http://nlp.stanford.edu/data/glove.twitter.27B.zip.\nThe output file is just a name for your output file.'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print './makeTwitterGloveFileCompatibleForGensim.py -i <input file> -o <output file>'
         print 'The input directory should contain all the training files. \nThe output directory will be where the models are stored.'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputFile = arg
      elif opt in ("-o", "--ofile"):
         outputFile = arg   
   return inputFile, outputFile


# In[25]:

def main(argv):
    inputFile, outputFile = getRelevantFiles(argv)
    #inputFile = "glove.twitter.27B.25d.txt"    
    numLines = getNumLines(inputFile)
    numDims = getNumDims(inputFile)
    print "Lines: %d\nDims: %d" % (numLines, numDims)
    
    tempFile = writeTempLengthFile(numLines, numDims)
    x = writeConvertedFile(inputFile, tempFile, outputFile)


# In[26]:

if __name__ == "__main__":
   main(sys.argv[1:])



