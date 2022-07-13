# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:01:17 2022

@author: blitt
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import regex as re
import numpy as np
import spacy
import seaborn as sns
from wordcloud import WordCloud
import pickle
import sys
from collections import Counter  
from transformers import pipeline

class Document: 

    def __init__(self, slidesCsv, nlp, **kwargs): 
        self.nlp = nlp
        self.dataFrame = self.formatSlides(slidesCsv)
        
        #a premade hugging-face/transformers pipeline that seems to work well
        #for sentiment classification 
        if "sentAnalysis" in kwargs and kwargs["sentAnalysis"] == True: 
            self.sentClassifier = pipeline("sentiment-analysis")  

        #cutoff for how many lemmas we care about when creating word clouds etc..
        #doesn't seem neccessary anymore... don't want to make this global 
        """
        self.cutoff = 5
        if "cutoff" in kwargs: 
            self.cutoff = kwargs["cutoff"]
        """
        colInfo = None 
        if "subsetColumns" in kwargs: 
            #column names that we want to get subsets for 
            self.subsetColumns = kwargs["subsetColumns"]
            
            #for each column, get rowName-wise information, store that in a dictionary 
            colInfo = {}
            doubleColInfo = {}

            #the desired behavior is that subsetColumns can have a string,
            #meaning we subset by an individual column, or an inner list 
            #meaning that we subset by two columns, in the order they are 
            #given 
            for columnName in self.subsetColumns: 
                if type(columnName) == str: 
                    #outermost dictionary: keys = column name, values = dictionary of rowName   
                    #dictionary of rowNames: keys = rowNames, values = dictionary of nlp items/lists/objects
                    #inntermost dictionary: keys = same name as properties of this dictionary, values = lists/objects etc..
                    colInfo[columnName] = self.initColumn(columnName)
                else: 
                    #remember in this case columnName is a LIST 
                    #so we are passing list of two cols ot function 
                    doubleColInfo[str(columnName[0]) + "," +  str(columnName[1])] = self.initColumns(columnName)
                
            self.colInfo = colInfo
            self.doubleColInfo = doubleColInfo 
         
        #we can load dictionaries of word/lemma frequencies to scale our counts by 
        if "freqLists" in kwargs: 
            freqLists = kwargs["freqLists"]
            if "COWWords" in freqLists: 
                #load pickled COW corpora word frequencies 
                #adds about 10 seconds to the load time 
                COWWordsPath = "/home/blitt/Academic/PSC/NLPPipelines/AG2PIAnalytics/COWWordFreqDict.pkl"
                with open(COWWordsPath, 'rb') as handle:
                    COWWordFreqs = pickle.load(handle)
                    self.COWWords = COWWordFreqs
                
                #constant
                #the median frequency of words in COW dataset 
                self.COWWordMedian = 2
                self.COWWordSum = 8217071085

            if "COWLemmas" in freqLists: 
                #load pickled COW corpora lemma frequencies 
                #adds about 10 seconds to the load time 
                COWLemmaPath = "/home/blitt/Academic/PSC/NLPPipelines/AG2PIAnalytics/COWLemmaFreqDict.pkl"
                with open(COWLemmaPath, 'rb') as handle:
                    COWLemmaFreqs = pickle.load(handle)
                    self.COWLemmas = COWLemmaFreqs
                #constant
                #the median frequency of lemmas in COW dataset 
                self.COWLemmaMedian = 2
                self.COWLemmaSum = 7575421206
                
        #add some functionality later to specify which cols we are interested in etc... 
        #default behavior is to init allTokens, importantTokens, and workingDoc 
        #using all tokens in the entire doc 
        self.initAllWordsPipe()
        
    def initAllWordsPipe(self): 
        noJunkChars = self.removeJunkChars(". ".join(self.dataFrame["value"]))
        
        #the smallest amount of cleaning possible 
        self.noJunkChars = noJunkChars
        
        #NOTE: we create a doc with stop words in because we need to get accurate parts of speech 
        self.workingDoc = self.nlp(noJunkChars)

        self.allTokens = [item for item in self.workingDoc]
        #this will return a list of tokens 
        onlyImportant = self.getImportantWordList(self.allTokens)
        
        self.importantTokens = onlyImportant
        
        self.importantLemmas = self.getLemmas(onlyImportant)
         
        self.spans = self.getSpans(self.workingDoc) 

    def initColumn(self, colName): 
        colInfo = {}
        df = self.dataFrame
        for rowName in df[colName].unique():
            
            #get only rows with this specific rowName
            subsetDf = df[df[colName] == rowName]
            
            #remove junkChars 
            noJunkChars = self.removeJunkChars(". ".join(subsetDf["value"]))
            
            #spaCy object
            workingDoc = self.nlp(noJunkChars)
            allTokens = [item for item in workingDoc]
            
            #this will return a list of tokens 
            onlyImportant = self.getImportantWordList(allTokens)

            importantLemmas = self.getLemmas(onlyImportant)
            
            #the same information/items that we have for the whole dataframe (same variable names)
            #but subsetted for this column and rowName subset
            subsetDict= {"dataFrame":subsetDf, "noJunkChars":noJunkChars, "workingDoc":workingDoc, "importantTokens":onlyImportant, "importantLemmas":importantLemmas}
            
            colInfo[rowName] = subsetDict
        return colInfo
    
    def initColumns(self, colNames): 
        colInfo = {}
        df = self.dataFrame

        #just grab column names from list (i.e. colNames is a list) 
        colOne = colNames[0]
        colTwo = colNames[1] 
        for rowOne in df[colOne].unique(): 
            innerDict = {}
            for rowTwo in df[colTwo].unique(): 
                #get only rows with the current combination of colNames that we
                #splitting by 
                subsetDf = df[(df[colOne] == rowOne) & (df[colTwo] == rowTwo)]
            
                #remove junkChars 
                noJunkChars = self.removeJunkChars(". ".join(subsetDf["value"]))
            
                #spaCy object
                workingDoc = self.nlp(noJunkChars)
                allTokens = [item for item in workingDoc]
            
                #this will return a list of tokens 
                onlyImportant = self.getImportantWordList(allTokens)

                importantLemmas = self.getLemmas(onlyImportant)
            
                #the same information/items that we have for the whole dataframe (same variable names)
                #but subsetted for this column and rowName subset
                subsetDict= {"dataFrame":subsetDf, "noJunkChars":noJunkChars, "workingDoc":workingDoc, "importantTokens":onlyImportant, "importantLemmas":importantLemmas}
                
                #the second layer of the dictionary will contain information
                #from a particular column, but subsetted by unique values of
                #the second column  
                innerDict[rowTwo] = subsetDict
            colInfo[rowOne] = innerDict
        return colInfo 

    #we want to extract the appropriate item from the nested dict
    #while keeping the nested dict structure 
    def getDoubleColSubAttributeDict(self,colList, attribute): 
        """
        example from the version that doesn't have nested dictionary (we are
        extracting when we have two columns 
        if self.colInfo != None: 
            colInfo = self.colInfo
            outDict = {}
            
            if column in colInfo: 
                for rowName in colInfo[column].keys(): 
                    #get column, row, and attribute info, store using only only rowName in outDict
                    outDict[rowName] = colInfo[column][rowName][attribute]
                return outDict
            else: 
                print("can't find specified column. Options are: " + str(colInfo.keys()))
        else: 
            print("columns not specified at instantiation (i.e. creation) of document object")
            return None
        """
        if self.doubleColInfo != None: 
            colInfo = self.doubleColInfo 
            outDict = {}
            
            mergedStr = colList[0] + "," + colList[1]
            if mergedStr in colInfo: 
                for rowOne in colInfo[mergedStr].keys(): 
                    outDict[rowOne] = {}
                    currDict = {}
                    for rowTwo in colInfo[mergedStr][rowOne].keys():
                        #pull out the attribute at this point in the dict and make it available at rowOne, rowTwo 
                        currDict[rowTwo] = colInfo[mergedStr][rowOne][rowTwo][attribute] 
                    outDict[rowOne] = currDict
                return outDict
            else: 
                print("can't find specified columns. Remember to type enter [colOneName, colTwoName]") 
                return None 
        else: 
            print("column pair not specified at instation (i.e. creation of document object)") 
            return None 
    #TODO: make this work for variable column names 
    #take the raw csv (copied and pasted into an excel file) and get it into a long format 
    #we will have columns for persona, the column in the original grid (variable), and the touchpoint in question
    def formatSlides(self, inDf):
        df = pd.read_csv(inDf, sep=",", names = ["Touchpoint", "Pre_Engagement", "Engagement", "Post_Engagement", "Persona"], header=0)
        
        #we want this data to be in LONG format, so melt columns to rows 
        longDf = df.melt(id_vars=["Persona", "Touchpoint"], value_vars=["Engagement", "Pre_Engagement", "Post_Engagement"])
        formattedDf = longDf.set_index(["Persona", "variable", "Touchpoint"]).apply(lambda x: x.str.split("\n").explode()).reset_index()
        formattedDf = formattedDf.rename(columns={"variable":"Engagement"})
        return formattedDf 
    
    #inText is a string 
    #returns a string but cleaned of chars we don't need
    def removeJunkChars(self, inText): 
        
        #remove all characters that aren't punctuation letters or numbers 
        allText = re.sub("[^a-zA-Z0-9 \.!\?']", "", inText)

        #replace all punctuation that has no words between it with period. Keep one space on right side
        allText = re.sub("(\s*(\.|!|\?)\s*)+", ". ", allText)

        return allText
    
    #takes a list of SPACY tokens and outputs a list of "important" words, i.e. words that 
    #contain meaning and aren't spaces or punctuation
    #note, these tokens are SPACY tokens, so they have all the right pos tagging etc
    def getImportantWordList(self, inList): 
        return [item for item in inList if item.is_stop == False and len(str(item).strip()) > 0 and item.pos_ != "PUNCT"]
    
    #get lemmas for a given list. Bear in mind we can't get lemmas if we don't have spacy objects to work with
    #AND once we get the lemmas they are no longer the spacy objects we may eventually need 
    def getLemmas(self, inList): 
        return [item.lemma_ for item in inList]
   
    #get spans from a spacy document object called inDoc
    #This is the helper since it only takes inDocs and we want to eventually be able to 
    #call getSpans for a dictionary of workingDocs 
    def getSpansHelper(self, inDoc): 
        #now we want to get spans on punctuation 
        spanList = []
        spanStart = 0 
        spanEnd = 0
        for token in inDoc: 
            if token.is_punct: 
                currSpan = inDoc[spanStart:spanEnd]
                spanList.append(currSpan)
                spanStart = spanEnd + 1
            spanEnd += 1
        return spanList
    
    #get spans for a document or dictionary of documents. Used during init to assign global var   
    def getSpans(self, inDoc): 
        if type(inDoc) != dict: 
            return self.getSpansHelper(inDoc) 

    ##################################################################################33
    #NOTE: start of pipeline to create wordcloud etc...
    
    #TODO: get started on some sort of frequencies that are scaled 
    #TODO: make all visualizations simply return the fig object
    #TODO: add a word cloud option to the 
    
    #get a palette (i.e. dictionary with colors as values) given a list of lists 
    #basically each word in the inner lists will have the same colors
    #joinWith tells us what to join words with when creating palette dicionary 
    #outputs a palette dictionary with joined phrases and 
    def getPalette(self, wordGroups, **kwargs): 
        joinWith = "_"
        if "joinWith" in kwargs: 
            joinWith = kwargs["joinWith"]
        palette = sns.husl_palette(len(wordGroups))
        rgbPal = []
        for item in palette: 
            newTup = tuple(int(item*255) for item in item)
            rgbPal.append(newTup)

        outDict = {}
        for i in range(len(wordGroups)): 
            group = wordGroups[i]
            for word in group: 
                outDict[joinWith.join(str(word).split())] = rgbPal[i]
        
        return outDict
    
    #returns [dictionary of counts, sorted list of words (by counts), sorted list of counts] where word freq is above given count
    #NOTE: inList must be a list of objects, or dictionary 
    #ALSO works with input dictionary where values are lists (stratified column values)
    #in this case returns dictionary where values tuple of what we normally return and keys are the same as input
    def getFreqDictCutoff(self, inList, cutoff):
        if type(inList) is dict: 
            #remember in this case inList is actually a DICTIONARY
            outDict = {}
            for key in inList.keys(): 
                countDict = Counter(inList[key])
                overCutoff = {k:v for k, v in countDict.items() if v > cutoff}
                sortedTokens = sorted(list(overCutoff.keys()), key=lambda x: overCutoff[x])
                sortedCounts = [overCutoff[item] for item in sortedTokens]
                toReturn = overCutoff
                outDict[key] = toReturn
            return outDict
        
        #create counter object 
        countDict = Counter(inList)
        overCutoff = {k:v for k, v in countDict.items() if v > cutoff}
        sortedTokens = sorted(list(overCutoff.keys()), key=lambda x: overCutoff[x])
        sortedCounts = [overCutoff[item] for item in sortedTokens]
        return overCutoff, sortedTokens, sortedCounts
        
    def getFreqDict(self, inList):
        return self.getFreqDictCutoff(inList, 0)
    
    #takes a frequency dictionary and returns a version that is "unzipped" 
    #meaning that it returns a list of tokens and a list of frequencies sorted by frequency 
    def unZipFreqDict(self, inFreqDict): 
        freqKeys = sorted(list(inFreqDict.keys()),key=lambda x: inFreqDict[x], reverse=True)
        freqVals = sorted(list(inFreqDict.values()), reverse=True)
        return (freqKeys, freqVals) 

    #Takes a list and outputs a dictionary containing the top itemNum # tokens from the inList as well as their frequencies
    #NOTE: we will need resort that dictionary at some point if using later for graphing etc... 
    #This is a helper because it can't handle a dictionary as input, so we can only use it for the raw lists of text/spacy objs 
    def getTopFewHelper(self,inList, itemNum):
        freqDict = self.getFreqDict(inList)[0]
        sortedTokens = sorted(list(freqDict.keys()), key=lambda x: freqDict[x], reverse=True)
        return {k:freqDict[k] for k in sortedTokens[:itemNum]}

    def getTopFew(self, inList, itemNum): 
        if type(inList) == list: 
            return self.getTopFewHelper(inList, itemNum)
    
    #for a given list of words or spacy tokens, get a dictionary of spans containing these words/tokens
    def getWordSpans(self, inList): 
        outDict = {}
        for span in self.spans: 
            for word in inList: 
                word = str(word) 
                if word in str(span): 
                    if word not in outDict: 
                        outDict[word] = [span]
                    else: 
                        outDict[word].append(span)
        return outDict

    #this is used to take out all duplicates from a list of spans
    #it outputs the exact same list of spans but with spaces at the front if the span is a duplicate 
    def spaceSpans(self, inSpans):
        outList = []
        spanDict = {}
        for span in inSpans:
            span = str(span)
            count = 0
            #keep adding spaces to item until it does not exist in span dict
            while count < len(inSpans):
                spacer = "".join([" " for temp in range(0, count)])
                newToken = spacer + str(span)
                if newToken not in spanDict:
                    spanDict[newToken] = True
                    outList.append(newToken)
                    break
                count +=1
        return outList

    #take input (iterable of spaCy tokens), lemmatize if not already lemmatized, generate 
    #lemmas should be a boolean true or false value for whether or not we scale with lemma dict
    #NOTE: log is used here to even out distribution 
    def getScaledFreqDictCutoffHelper(self, inList, lemmas, cutoff): 
        if lemmas == True:  
            freqDict = self.COWLemmas 
            medianFreq = self.COWLemmaMedian
            totalOccurences = self.COWLemmaSum 
        else: 
            freqDict = self.COWWords 
            medianFreq = self.COWWordMedian
            totalOccurences = self.COWWordSum 
            
        #create counter object
        countDict = Counter(inList)
        
        #how many times more or less frequent 
        scaledCounts = {}
        notAvail = 0 
        for token,count in countDict.items(): 
            if token in freqDict:
                corporaFreq = freqDict[token]
                #prob of token in our list / prob of token in reference corpora
                #the log will help even out the distribution 
                scaledCounts[token] = np.log((countDict[token]/sum(list(countDict.values())))  /  (corporaFreq / totalOccurences))
            else: 
                notAvail += 1
                #TODO figure out how to deal with tokens that aren't in our corpora 
                #scaledCounts[token] = (countDict[token]/len(inList)) / (medianFreq/corporaFreq) 
        print(str(notAvail) + " not found in corpora")
        
        overCutoff = {k:v for k,v in scaledCounts.items() if v > cutoff}
        sortedTokens = sorted(list(overCutoff.keys()), key=lambda x: overCutoff[x])
        sortedCounts = [overCutoff[item] for item in sortedTokens]
        return overCutoff, sortedTokens, sortedCounts
    
    def getScaledFreqDictCutoff(self, inList, lemmas, cutoff): 
        #we want to be able to handle the case where we have a dictionary where the keys are 
        #unique row values in a colummn and the values are lists of spaCy tokens 
        if type(inList) == dict: 
            outDict = {}
            
            #just perform the function on every list in the values of the dictionary 
            for key,value in inList.items(): 
                outDict[key] = self.getScaledFreqDictCutoffHelper(value, lemmas, cutoff)[0]
            return outDict
        else: 
            return self.getScaledFreqDictCutoffHelper(inList, lemmas, cutoff)
        
    def FreqDictBarChart(self, inDict, **kwargs): 
        sortedTokens = sorted(list(inDict.keys()), key=lambda x: inDict[x])
        sortedCounts = [inDict[item] for item in sortedTokens] 
        currFig = plt.barh(sortedTokens, sortedCounts, **kwargs)
        
        return currFig
    
    def sentBarChart(self, sentDict): 
        #unpack dictionary into two sorted lists
        tokens = sorted(list(sentDict.keys()), key=lambda x: sentDict[x], reverse=False)
        sents = sorted(list(sentDict.values()), reverse=False)

        green = "#41ae76"
        red = "#ef6548"
        colorList = [green if sent > 0 else red for sent in sents]
        legendElements = []
        legendElements.insert(0, Patch(facecolor=red,label="Negative"))
        legendElements.insert(0, Patch(facecolor=green,label="Positive"))

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        plt.legend(handles=legendElements,loc='lower right')

        ax.barh(tokens, sents, color=colorList)
        return ax 

    #create a bar chart using a dictionary of frequencies and a dictionary of word sentiments 
    def FreqSentBarChart(self, freqDict, sentDict): 
        sortedTokens = sorted(list(freqDict.keys()), key=lambda x: freqDict[x])
        sortedSents = [sentDict[item] for item in sortedTokens] 
        sortedCounts = [freqDict[item] for item in sortedTokens] 
        greens = ["#99d8c9", "#66c2a4", "#41ae76","#238b45","#005824"]  
        reds = ["#990000", "#d7301f", "#ef6548", "#fc8d59", "#fdbb84"]
        colorSpectrum = reds + greens 
        barColors = []
        for item in sortedTokens: 
            itemSent = float(sentDict[item]) 
            itemColor = "" 
            for index, currCutoff in enumerate(np.arange(-.8, 1.01, .2)): 
                #if we don't already have a color and are below the cutoff for the first time 
                #assign a color and add to barColor list 
                if itemColor == "" and itemSent <= currCutoff:
                    itemColor = colorSpectrum[index]
                    barColors.append(itemColor) 
        legendElements = []
        legendElements.insert(0, Patch(facecolor="#41ae76",label="Positive"))
        legendElements.insert(0, Patch(facecolor="#ef6548",label="Negative"))
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        plt.legend(handles=legendElements,loc='best')
        ax.barh(sortedTokens, sortedCounts, color=barColors)
        return ax 

    #TODO: create a legend so we know which bars to to which row names 
    def FreqDictBarChartColumn(self, inDict, **kwargs): 
        allTokens = []
        allCounts = []
        
        #get palette with number of colors equal to num keys in dictionary  
        palette = sns.husl_palette(len(inDict.keys()))
        
        #needed to prevent same words from being combined 
        spacer = ""
        
        colorList = []
        legendElements = []
        
        #NOTE: this sorting is what will keep things consistent when creating word cloud 
        keys = sorted(list(inDict.keys()))
        
        #NOTE: DELETE
        print(keys)
        
        #NOTE: DELETE
        print(palette)
        
        for count in range(len(keys)): 
            keyName = keys[count]
            thisDict = inDict[keyName]
            
            sortedTokens = sorted(list(thisDict.keys()), key=lambda x:thisDict[x])
            sortedCounts = [thisDict[item] for item in sortedTokens]
            
            #add to larger list 
            allTokens += [spacer + item for item in sortedTokens]
            allCounts += sortedCounts
            
            currColor = palette[count]
            colorList += [currColor for i in range(len(sortedTokens))]
            
            legendElements.insert(0, Patch(facecolor=currColor,label=keyName))
            
            spacer += " "
        fig = plt.figure()
        plt.barh(allTokens, allCounts, color=colorList)

        plt.legend(handles=legendElements,loc='best')
        return fig
    
    #borrowed heavily from: https://stackoverflow.com/questions/42191668/matplotlib-dynamic-bar-chart-height-based-on-data/42192931 
    #barwidth = inch per bar
    #spacing = spacing between subplots in units of barwidth
    #figx = figure width in inch
    #left = left margin in units of bar width
    #right = right margin in units of bar width
    def FreqDictBarChartColumnSplit(self, inDict, barwidth, spacing, figx, top, bottom, left, right ,**kwargs):      
        #the total number of subplots we will have 
        tc = len(inDict.keys())
        print("num of cats: " + str(tc))

        #the subplots need to be in a certain order so the coloring system is correct
        keys = sorted(list(inDict.keys()))
        
        #get palette with number of colors equal to num keys in dictionary  
        palette = sns.husl_palette(len(inDict.keys()))
                
        max_values = []  # holds the maximum number of bars to create
        for key in keys:
            max_values.append(len(list(inDict[key].values())))
        max_values = np.array(max_values)
        print("max values array: " + str(max_values)) 

        # total figure height:
        #tried adding one more bar to put main title on 
        figy = ((np.sum(max_values)+tc) + (tc+1)*spacing)*barwidth #inch

        #add any padding we may need for extra figure level labels  
        paddedY = figy + (top+bottom)*barwidth 
        
         
        fig = plt.figure(figsize=(figx,paddedY))
        ax = None
        
        #a counter, called index to work with copied code
        #TODO: take a look at bug where last line has issue with its title/padding
        index = 0  
        for key in keys:
            currDict = inDict[key]
            #values to go on y axis as labels
            entries = list(currDict.keys())
            #the length of the bars (term counts) 
            values = [currDict[item] for item in entries]
           
            y_ticks = range(1, len(entries) + 1)
           
            # coordinates of new axes [left, bottom, width, height]
            # how to add in top and bottom margin?
            #a very hack sol'n to error where last plot doesn't have enough padding for some reason
            if index == len(keys)-1:
                extraSpace = top*barwidth
            else: 
                extraSpace = 0
            
            coord = [left*barwidth/figx, 
                     (1-barwidth*((index+1)*spacing+np.sum(max_values[:index+1])+index+1 +top + extraSpace)/paddedY),  
                     1-(left+right)*barwidth/figx,  
                     (max_values[index]+1)*barwidth/figy ] 

            ax = fig.add_axes(coord, sharex=ax)
            ax.barh(y_ticks, values, color=palette[index])
            ax.set_ylim(0, max_values[index] + 1)  # limit the y axis for fixed height
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(entries)
            ax.invert_yaxis()
            ax.set_title(key, loc="left")
            index += 1
            """        
            #NOTE: this sorting is what will keep things consistent when creating word cloud 
            keys = sorted(list(inDict.keys()))
            
            #NOTE: potentially add default figure size?
            #get axes equal to number of keys in the inDictionary 
            if "figsize" not in kwargs: 
                print("specify figure dimensions with figsize = (dim1 ,dim2)")
                sys.exit() 
            else: 
                pFigsize = kwargs["figsize"]
            #make sure bar width is specified as well 
            if "width" not in kwargs: 
                print("specify bar width for subplots by specifying width = barWidth") 
                sys.exit()
            else: 
                pWidth = kwargs["width"]

            fig, ax = plt.subplots(len(palette), 1, figsize=pFigsize)
            for i in range(0, len(keys)):
                currKey = keys[i]
                thisDict = inDict[currKey]
     
            #ax[i, 1].barh(inDict.)    
#            print(inDict[currKey].items())
        
            #ax[i].barh(inDict[currKey])       
             
            sortedTokens = sorted(list(thisDict.keys()), key=lambda x:thisDict[x])
            sortedCounts = [thisDict[item] for item in sortedTokens]
            
            ax[i].barh(sortedTokens, sortedCounts, )
            return (fig, ax) 
            """
        return (fig, ax) 
 
    #we sort the tokens before zipping to our 
    def ColoredFreqDictBarChart(self, inDict, **kwargs): 
        sortedTokens = sorted(list(inDict.keys()), key=lambda x: inDict[x])
        sortedCounts = [inDict[item] for item in sortedTokens]
        
        #get palette with number of colors equal to num of tokens 
        palette = sns.husl_palette(len(sortedTokens))
        
        #used to assign colors consistently
        colorDict = dict(zip(sortedTokens, palette))
        
        #plot tokens, counts sorted according to tokens, and with colors assigned to sorted tokens 
        toReturn = plt.barh(sortedTokens, sortedCounts, color=[colorDict[token] for token in sortedTokens], **kwargs)
        
        return toReturn 
    
    #take us to the color space needed for wordcloud 
    def toWordCloudColors(self, inPalette): 
        outPalette = []
        for item in inPalette: 
            newTup = tuple(int(item*255) for item in item)
            outPalette.append(newTup)
        return outPalette
    
    def CreateWordCloud(self, inDict, **kwargs): 
        
        #set up some defaults that can be changed if input into key word args when function is called 
        #we can add an avoidList to remove certain words 
        if "avoidList" in kwargs: 
            toAvoid = kwargs["avoidList"]
        else: 
            toAvoid = []

        #what item should we be joining our words with 
        #dash by default 
        if "joinWith" in kwargs:
            joinWith = kwargs["joinWith"]
        else: 
            joinWith = "_"
        
        wc = WordCloud(background_color="#f0f2f5", height=500, width=800).generate_from_frequencies(inDict)
        
        #get tokens from inDict (keys), sort them according to their freq
        #the sorting keeps things consistent with frequency dict bar chart
        sortedTokens = sorted(list(inDict.keys()), key=lambda x: inDict[x])
        
        #get palette with number of colors equal to num of tokens 
        #zip to tokens
        #color with dictionary of zip
        palette = self.toWordCloudColors(sns.husl_palette(len(sortedTokens)))
        print(palette)
        paletteDict = dict(zip(sortedTokens, palette))
        wc.recolor(color_func=lambda word, **kwargs: paletteDict[word])
        
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
     
        
    #we take a dictionary of dictionaries as input and create word cloud colored according to 
    #the different groups in the dictionary as output 
    def CreateColumnWordCloud(self, inDict, **kwargs): 
        #set up some defaults that can be changed if input into key word args when function is called 
        #we can add an avoidList to remove certain words 
        if "avoidList" in kwargs: 
            toAvoid = kwargs["avoidList"]
        else: 
            toAvoid = []

        #what item should we be joining our words with 
        #dash by default 
        if "joinWith" in kwargs:
            joinWith = kwargs["joinWith"]
        else: 
            joinWith = "_"
            
        #get number of columns equal to the number of keys in input dictionary 
        palette = self.toWordCloudColors(sns.husl_palette(len(list(inDict.keys()))))
        
        #we want to sort the dictionary keys list and use that as ordering, to stay consistent with bar chart
        #color scheme 
        colorDict = {}
        entireDict = {}
        sortedKeys = sorted(list(inDict.keys()))
        
        #NOTE: DELETE
        print(sortedKeys)
        
        #NOTE: 
        print(palette)
        
        for i in range(0, len(sortedKeys)): 
            
            #get dictionary of tokens and counts, iterate through it
            subDict = inDict[sortedKeys[i]]
            for token, freq in subDict.items(): 
                #so we don't need to worry about doubles 
                #keep searching for a number of padded tokens that needs to be added to prevent overwriting doubles
                count = 0
                while count < len(sortedKeys): 
                    spacer = "".join([" " for temp in range(0, count)]) 
                    newToken = str(token) + spacer 
                    if newToken not in colorDict: 
                        colorDict[newToken] = palette[i]
                        entireDict[newToken] = freq
                        break
                    count +=1 
        print(entireDict)
        #create wordcloud 
        wc = WordCloud(background_color="#f0f2f5", height=500, width=800).generate_from_frequencies(entireDict)
        
        #color according to which key this word belonged to in input dictionary 
        wc.recolor(color_func=lambda word, **kwargs: colorDict[word])
        
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
               
        
    #extract only one attribute (i.e. lemmas etc..) and put it in dictionary stratified by unique row names
    def getColSubAttributeDict(self, column, attribute): 
        if self.colInfo != None: 
            colInfo = self.colInfo
            outDict = {}
            
            if column in colInfo: 
                for rowName in colInfo[column].keys(): 
                    #get column, row, and attribute info, store using only only rowName in outDict
                    outDict[rowName] = colInfo[column][rowName][attribute]
                return outDict
            else: 
                print("can't find specified column. Options are: " + str(colInfo.keys()))
        else: 
            print("columns not specified at instantiation (i.e. creation) of document object")
            return None
   
    #take a word -> spanList dictionary as input and output word -> sentiment dictionary 
    def getWordSpanSent(self, inDict): 
        outDict = {}
        for word, spanList in inDict.items(): 
            sents = []
            for span in spanList: 
                classified = self.sentClassifier(str(span))[0]
                if classified["label"] == "NEGATIVE": 
                    sent = -float(classified["score"])
                else: 
                    sent = float(classified["score"])
                sents.append(sent)
            outDict[word] = np.mean(sents)
        return outDict

    #takes spans, creates a dataframe with scores and pos/neg label as integer
    #then returns the top and bottom "numExamples" number of rows sorted by sentiment score 
    #example usage would be topDf, bottomDf = docObj.getExtremes(spanList, numExamples)
    def getExtremes(self, spanList, numExamples): 
        #it's better to classify entire list (faster) then unpack rather than classify 
        #one at a time 
        spanSents = self.sentClassifier([str(item) for item in spanList])

        #the classification (pos, neg) as an int (1, 0)
        spanClasses = [int(item["label"] == "POSITIVE") for item in spanSents]

        #the classification (pos, neg) as an int (1, 0)
        spanScores = [float(item["score"]) if item["label"] == "POSITIVE" else -float(item["score"]) for item in spanSents]

        #NOTE: many of the sents seem to be fairly close to 1
        spanDf = pd.DataFrame({"spans":self.spaceSpans(spanList), "scores":spanScores, "groundTruth":spanClasses})
        spanDf = spanDf.sort_values("scores")
        dfLen = len(spanDf)
        #return the top (positive) numExamples , the bottom (negative) numExamples
        return (spanDf.iloc[0:numExamples,:], spanDf.iloc[dfLen-numExamples:dfLen])

