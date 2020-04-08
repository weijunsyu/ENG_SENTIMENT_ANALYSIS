import numpy as np
import pandas as pd
import os.path

#Raw data filenames and folders
TWITTER_DATABASE_BINARY_SENTIMENT = "../Data/Raw_Data/Twitter_Sentiment_Data/training.1600000.processed.noemoticon.csv"

MOVIELENS_DATABASE = "../Data/Raw_Data/Movie_Lens_Data/ml-latest/"

REVIEW_POLARITY_POS_FOLDER = "../Data/Raw_Data/review_polarity/txt_sentoken/pos"
REVIEW_POLARITY_NEG_FOLDER = "../Data/Raw_Data/review_polarity/txt_sentoken/neg"

#Cleaned data filenames
TWITTER_BINARY_SENTIMENT_CLEAN = "../Data/Binary/Twitter_Sentiment.csv"
REVIEW_POLARITY_SENTIMENT_CLEAN = "../Data/Binary/Review_Sentiment.csv"

def GetDatasets(directory, fileExtension='auto', delimiter='None'):
    """
    Assumes all files in directory are text files (flat files) such as txt or csv files.
    If file extension is auto then all data in director will be processed.
    If file extension is given then only files of that specified extension will be processed.
    """
    datasets = []

    for file in os.listdir(os.fsencode(directory)):
        filename = os.fsdecode(file)
        dataset = ProcessDatasetType(fileExtension, filename, directory, delimiter)
        files.append(dataset)

    return datasets


def ProcessDatasetType(fileExtension, filename, directory, delimiter):
    """
    """
    if ((fileExtension == 'auto') or (filename.endswith(fileExtension))):
        filePath = os.path.join(directory, filename)
        dataset = pd.read_csv(filePath, sep=delimiter, engine='python')
        return dataset



if __name__ == "__main__":

    if os.path.isfile(TWITTER_BINARY_SENTIMENT_CLEAN):
        print("Cleaned data file: '" + TWITTER_BINARY_SENTIMENT_CLEAN + "' already exists")
    else:
        try:
            #read file
            sentimentDataTwitter = pd.read_csv(TWITTER_DATABASE_BINARY_SENTIMENT, sep=None, engine='python')
            #take the first column as y and the last column as the x
            cleaned_data = sentimentDataTwitter.iloc[:,[0,-1]]

            #Save file
            cleaned_data.to_csv(TWITTER_BINARY_SENTIMENT_CLEAN, sep=',', header=False, index=False, index_label=None, quoting=1)

        except FileNotFoundError:
            print("Raw data file: '" + TWITTER_DATABASE_BINARY_SENTIMENT + "' does not exist")

    if os.path.isfile(REVIEW_POLARITY_SENTIMENT_CLEAN):
        print("Cleaned data file: '" + REVIEW_POLARITY_SENTIMENT_CLEAN + "' already exists")
    else:
        try:
            #create and open new file to be made:
            cleanReview = open(REVIEW_POLARITY_SENTIMENT_CLEAN, 'a')

            #read each file in folder for positive sentiment and label as 0
            for file in os.listdir(os.fsencode(REVIEW_POLARITY_POS_FOLDER)):
                filename = os.fsdecode(file)
                filePath = os.path.join(REVIEW_POLARITY_POS_FOLDER, filename)
                filePosReview = open(filePath, "r") #open file
                content = filePosReview.read() #read from file
                content = content.replace('\n', '') #remove all newlines so entire text is one line
                content = content.replace('\"', '') #remove all double quotation marks from text
                content = content.replace('\'', '') #remove all single quotation marks from text
                cleanReview.write('\"0\",') #add label to cleaned file
                cleanReview.write('\"') #add quotation marks indicating string input
                cleanReview.write(content) #add content
                cleanReview.write('\"') #add ending quotes
                cleanReview.write('\n') #add newline indicating end of data tuple
                filePosReview.close() #close raw data file

            #read each file in folder for negative sentiment and label as 1
            for file in os.listdir(os.fsencode(REVIEW_POLARITY_NEG_FOLDER)):
                filename = os.fsdecode(file)
                filePath = os.path.join(REVIEW_POLARITY_NEG_FOLDER, filename)
                fileNegReview = open(filePath, "r") #open file
                content = fileNegReview.read() #read from file
                content = content.replace('\n', '') #remove all newlines so entire text is one line
                content = content.replace('\"', '') #remove all double quotation marks from text
                content = content.replace('\'', '') #remove all single quotation marks from text
                cleanReview.write('\"1\",') #add label to cleaned file
                cleanReview.write('\"') #add quotation marks indicating string input
                cleanReview.write(content) #add content
                cleanReview.write('\"') #add ending quotes
                cleanReview.write('\n') #add newline indicating end of data tuple
                fileNegReview.close() #close raw data file

            cleanReview.close() #close cleaned data file

        except FileNotFoundError:
            print("Raw data file: '" + __FILE__ + "' does not exist")































#end
