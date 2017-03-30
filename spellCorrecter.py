#!/usr/bin/envpython
from __future__ import division
from collections import Counter
import re
import os
import sys
import math

unigramDict = {}    # dictionary of the unigrams in the spell-errors document        
bigramDict = {}     # dictionary of the bigrams in the spell-errors document
#The confusion dictionaries represent the confusion matrices in the noisy channel model
insConfusion = {}   # dictionary of bigrams written as xy instead of x, so, the insertions  
delConfusion = {}   # dictionary of bigrams written as x instead of xy, so, the deletions
transConfusion = {} # dictionary of bigrams written as yx instead of xy, so, the transposition
subConfusion = {}   # dictionary of bigrams written as y instead of x, so, the substitution


def words(text): 
    """ Reads the words from the text"""
    return re.findall(r'\w+', text.lower())


def createConfusionDictsByFindingEdits(original, misspelled,numberOfOccurrence):
    """
    Gets the correct word and the misspelled word and its occurrence in the spell-errors document.
    Calculates the Damerau Levenshtein edit distance and finds the edit types by backtracing.
    Forms the confusion dictionaries based on these edits found. 
    """
    
    if original == misspelled: 
        
        return True
    
    else:
        
        lenOforig = len(original)
        lenOfmis = len(misspelled)
        
        dist = [[0 for i in range(lenOfmis+1)] for j in range(lenOforig+1)]         # represents edit distance matrix. 
        backtrace = [[' ' for x in range(lenOfmis+1)] for y in range(lenOforig+1)]  # contains information about which edit type is chosem.
        backtrace[0][0]='N'             # 'N' means no edit is required
        
        for i in range(lenOforig+1):
        
            dist[i][0] = i
            backtrace[i][0] = 'D'       # 'D' is used for delete operation
        
        for i in range(lenOfmis+1):
        
            dist[0][i] = i
            backtrace[0][i] = 'I'       # 'I' represents insertion  

        for i in range(1,lenOforig+1):
        
            for j in range(1,lenOfmis+1):
                
                # Compute the minimum of costs of deletion, insertion and substitution
                if original[i-1] == misspelled[j-1]:    # no operation
                    
                    dist[i][j] = dist[i-1][j-1]
                    backtrace[i][j]='N'
                
                else:
                    
                    delCost = dist[i-1][j] + 1                    
                    insCost = dist[i][j-1] + 1
                    
                    # Compute deletion cost
                    if delCost <= insCost:
        
                        dist[i][j] = delCost
                        backtrace[i][j] = 'D'
                    
                    # Compute insertion cost
                    else:

                        dist[i][j] = insCost
                        backtrace[i][j] = 'I'    
                    
                    # Compute substitution cost
                    subCost = dist[i-1][j-1]+1
                    if subCost < dist[i][j]:
                        
                        dist[i][j] = subCost
                        backtrace[i][j] = 'S'           # 'S' is for substitution
                
                # Check if transposition is possible and compute its cost
                if (i > 1) and (j > 1) and (original[i-1] == misspelled[j-2]) and (original[i-2] == misspelled[j-1]):
                    
                    transCost = dist[i-2][j-2] + 1
                    if transCost < dist[i][j]:
                        
                        dist[i][j] = transCost
                        backtrace[i][j] = 'T'           # 'T' represents transposition

        row = lenOforig 
        col = lenOfmis                
        # Determine edits by backtracing the operations stored in backtrace matrix
        # Add the edits to the corresponding confusion dictionaries
        while row > 0 and col > -1:
        
            if backtrace[row][col] == 'I':              # firstChar typed as firstChar+secondChar

                if row > 0:
                    
                    firstChar = original[row-1]
                
                else:
                    
                    firstChar = "@"                     # means that firstChar is empty  
                
                secondChar = misspelled[col-1] 
                bigram = firstChar + secondChar
                
                if bigram not in insConfusion:
                 
                    insConfusion[bigram] = 1
                
                else:
                    
                    insConfusion[bigram] = insConfusion[bigram] + 1 * numberOfOccurrence    # Adds the number of occurrence of the bigram to corresponding confusion dictionary. 
                
                col = col - 1
                    
            elif backtrace[row][col] == 'D':            # firstChar+secondChar typed as firstChar
                
                if row > 1:

                    firstChar = original[row-2]
                
                else:
                    
                    firstChar = "@"
                
                secondChar = original[row-1] 
                bigram = firstChar + secondChar
                
                if bigram not in delConfusion:
                
                    delConfusion[bigram] = 1
                
                else:
                
                    delConfusion[bigram] = delConfusion[bigram] + 1 * numberOfOccurrence
                
                row = row - 1
               
            elif backtrace[row][col] == 'T':            # firstChar+secondChar typed as secondChar+firstChar
                
                firstChar = original[row-2]
                secondChar = original[row-1]
                bigram = firstChar + secondChar
                
                if bigram not in transConfusion:
                
                    transConfusion[bigram] = 1
                
                else:
                
                    transConfusion[bigram] = transConfusion[bigram] + 1 * numberOfOccurrence
                
                row = row - 2
                col = col - 2
             
            else:
                
                if backtrace[row][col] == 'S':          # firstChar typed as secondChar in misspelled word
                    
                    firstChar = original[row-1] 
                    secondChar = misspelled[col-1]
                    bigram = firstChar + secondChar                                
                    
                    if bigram not in subConfusion:
                    
                        subConfusion[bigram] = 1
                    
                    else:
                    
                        subConfusion[bigram] = subConfusion[bigram] + 1 * numberOfOccurrence
               
                row = row -1 
                col = col -1        
  
        return True   
    
def getSpellErrors(filepath):
    """ 
    Gets and tokenizes the words from spell-errors document
    Creates unigram and bigrams using them
    Calls createConfusionDictsByFindingEdits() method with arguments: original word, misspelled versions of it and their occurrrences
    """
    file = open(filepath,'r')
    
    for line in file:
        
        parts = re.split(r'\W\s',line.lower())
        parts[-1] = parts[-1].strip()           # removes the \n from the last word
        createNgrams(parts[0])                  # creates unigrams and bigrams from the original word
        numberOfOccurrence = 1
        for i in range(1,len(parts)):
            createNgrams(parts[i])
            misspelled = parts[i]
            pairNoWord = re.split(r'(\d)', misspelled) # consists of the misspelled word and its number of occurrence
            
            if len(pairNoWord) > 1:
            
                misspelled = pairNoWord[0].strip()
                numberOfOccurrence = pairNoWord[1]
            
            createConfusionDictsByFindingEdits(parts[0], misspelled, int(numberOfOccurrence))
    
    #print unigramDict
    #print bigramDict    
    file.close()      

def createNgrams(word):
    
    "Creates unigrams and bigrams using 'word'"
    
    if "@" not in unigramDict: # adds @ for empty 
        
        unigramDict["@"]=1
    
    else:   
        
        unigramDict["@"]=unigramDict["@"] + 1
    
    prevChar = "@"
    
    for currChar in word:
        
        if currChar not in unigramDict:
            
            unigramDict[currChar] = 1
        
        else:
            
            unigramDict[currChar] = unigramDict[currChar] + 1
        
        bigram = prevChar + currChar    
        
        if bigram not in bigramDict:
            
            bigramDict[bigram] = 1
        
        else:   
            
            bigramDict[bigram] = bigramDict[bigram] + 1
        
        prevChar = currChar

def processTestSetForNoisyChannel(infile, outfile):
    
    "Gets misspelled words and find corrections by using noisy channel model"
    
    inf = open(infile,'r')
    outf = open(outfile,'w')
    
    for line in inf:
        
        line = line.strip()
        correctWord = mostProbableByOneEditDist(line)
        
        if correctWord == None:
        
            correctWord = ""
        
        outf.write(str(correctWord)+"\n")

    outf.close()
    inf.close()    
           
def processTestSetForLanguageModel(infile,outfile):
    
    "Gets misspelled words and find corrections by using language model"
    
    inf = open(infile,'r')
    outf = open(outfile,'w')
    
    for line in inf:
    
        line = line.strip()
        correctWord = correction(line)
    
        if correctWord == None:
            
            correctWord = ""
        
        outf.write(str(correctWord)+"\n")

    outf.close()
    inf.close()     

def mostProbableByOneEditDist(word):

    """
    Finds all edits that are one edit away from `word`.
    Computes their probabilities and decide the most probable one
    """
    
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    mostProbable = (None,0) # represents most probable word and its probability

    for L, R in splits:
        
        # Finds the candidates by distance with one insertion and calculates its probability    
        if R:

            correct = L + R[1:]
            firstChar1 = L[-1] if len(L)>0 else ""
            firstChar2 = L[-1] if len(L)>0 else "@"
            secondChar = R[0] 
            bigram = firstChar1 + secondChar
            insValue = insConfusion[bigram] if firstChar1 + secondChar in insConfusion else 0 # the value of the bigram in insConfusion
            uniValue = unigramDict[firstChar2] if firstChar2 in unigramDict else 0 # the number of times the firstChar2 is in unigramDict
            probability = (insValue + 1 ) / (uniValue + len(letters)) * P(correct) # the probability of the candidate = P(t|c)*P(c)
    

            if probability > mostProbable[1]:
            
                mostProbable = (correct, probability)
            #print mostProbable
    for L, R in splits:
        # Finds the candidates by distance with one transposition and calculates its probability      
        if len(R)>1:

            correct = L + R[1] + R[0] + R[2:]
            bigram = R[1] + R[0]
            transValue = transConfusion[bigram] if bigram in transConfusion else 0
            biValue = bigramDict[bigram] if bigram in bigramDict else 0
            probability = (transValue + 1 ) / (biValue + len(letters)) * P(correct)
            
            if probability > mostProbable[1]:
            
                mostProbable = (correct, probability) 
            #print mostProbable
    for L, R in splits:
        # Finds the candidates by distance with one substitution and calculates its probability      
        
        if R:
            
            for c in letters:
    
                correct = L + c + R[1:]
                bigram = c + R[0]
                subValue = subConfusion[bigram] if bigram in subConfusion else 0
                probability = (subValue + 1 ) / (unigramDict[c] + len(letters)) * P(correct)
               
                if probability >mostProbable[1]:
                    
                    mostProbable = (correct, probability)
                #if word == "assigmments":
                #    print mostProbable
                
    for L, R in splits:
        # Finds the candidates by distance with one deletion and calculates its probability                  
        for c in letters:
    
            correct = L + c + R 
            bigram = L[-1] + c if len(L)>0 else "@"+c
            delValue = delConfusion[bigram] if bigram in delConfusion else 0
            biValue = bigramDict[bigram] if bigram in bigramDict else 0
            probability = (delValue + 1 ) / (biValue + len(letters)) * P(correct)

            if probability > mostProbable[1]:

                
                mostProbable = (correct, probability)
          
            #print mostProbable 
     
    return mostProbable[0]

def P(word):

    "Probability of `word`"
    
    return ( WORDS[word] + 0.0001 )  / numberOfWords 

def correction(word): 

    "Most probable spelling correction for word based on language model."

    return max(candidates(word), key=P)

def candidates(word): 

    "Generate possible spelling corrections for word."

    return (known([word]) or known(edits1(word)) or [word])

def known(words): 

    "The subset of `words` that appear in the dictionary of WORDS."

    return set(w for w in words if w in WORDS)

def edits1(word):

    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    substitutes   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + substitutes + inserts)

 
def accuracy(computedWords, correctWords, modelType):
    
    "Calculates the accuracy of the model by using the computed words and the correct words"
    
    comp = open(computedWords)
    corr = open(correctWords)
    total = 0;
    correct = 0;
    
    for line in comp:
        
        line = line.strip()
        correctWord = corr.readline().strip()
        
        if line == correctWord:
        
            correct = correct + 1

        total = total + 1
    print "The correct ones:" , correct
    print "The wrong ones:", total - correct
    print "Accuracy of", modelType, (correct/total)*100 , "%"       



#corpus = sys.argv[1]   

#spellErrors = sys.argv[2]
corpus = "/corpus.txt"   

spellErrors = "/spell-errors.txt"

misspelledWords = sys.argv[1]
correctWords = sys.argv[2]
model = sys.argv[3] 
WORDS = Counter(words(open(os.getcwd()+corpus).read())) # reads the corpus and computes the number of each word 
numberOfWords = sum(WORDS.values()) # finds total number of words in the corpus
for word in WORDS:
    createNgrams(word)
getSpellErrors(os.getcwd()+spellErrors)
if(model == "noisychannel"):
    processTestSetForNoisyChannel(os.getcwd() +"/" +misspelledWords, os.getcwd() + "/noisy-channel-results.txt")
    accuracy(os.getcwd() + "/noisy-channel-results.txt", os.getcwd() + "/"+ correctWords, "Noisy Channel Model:")
elif(model == "language"):    
    processTestSetForLanguageModel(os.getcwd() +"/" +misspelledWords, os.getcwd() + "/language-results.txt")
    accuracy(os.getcwd() + "/language-results.txt", os.getcwd() +"/"+ correctWords, "Language Model:")
else:
    print "usage: python spellCorrecter.py <pathToTestMisspelledWords> <pathToTestCorrectWords> <model>"
#mostProbableByOneEditDist('bulletings')