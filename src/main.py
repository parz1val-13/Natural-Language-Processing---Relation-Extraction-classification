import argparse
import csv
import nltk
from nltk.corpus import stopwords
import re
import numpy as np
#import heapq
import math
import pandas as pd


def naive_bayes_classifier(doc,bow,bow_class):
    log_prior = {}   #stores log prior values
    big_doc = []     #stores big document values
    loglikelihood = {}
    classes = []
    total = 0

    for key,value in doc.items():
        i =  value.split(",")
        total += len(i)   
        classes.append(key)
    #print(classes)
    for c in classes:
        val = doc[c].split(',')
        n_doc = total
        n_c = len(val)
        logpc = math.log(n_c/n_doc)  #calculates log prior values
        log_prior[c] = logpc
        v= []
        for d in bow.keys():
            v.append(d)
        big_doc = doc
        #print(big_doc)
        loglike = []
        for w in range(len(v)):
            count_wc = 0
            count_wprime = 0
            #calculating values to be used in loglikelihood calculations
            for key,value in bow_class[c].items():
                if v[w] in bow_class[c]:
                    if v[w] == key:
                        count_wc = value
                    else:
                        count_wprime += value +1
                else:
                    count_wprime += 1

            loglikelihood_val = math.log((count_wc +1)/(count_wprime))  # calulates loglikelihood values
            loglike.append((v[w],loglikelihood_val))
        loglikelihood[c] = loglike

    return log_prior,loglikelihood,v,classes
    
def test_naive_bayes(testdoc,log_prior,loglikelihood,bow_classes,v,classes):
    #testing our classifier
    sum_c = {}
    #test_doc = []
    test_doc = testdoc.split()

    for c in classes:
        sum_c[c] = log_prior[c]
        for i in range(len(test_doc)):
            word = test_doc[i]
            
            if word in v:
                #loops through the loglikelihood dictionary which has classes as key and a list of tuples as value
                for x in loglikelihood[c]:
                    if word == x[0]:
                        #print(word,x[0])
                        value = x[1]
                        sum_c[c] = sum_c[c] + value
                        

    max_val = max(sum_c, key= sum_c.get)
    return max_val

# This function accepts input directory path as an argument,
# Opens the file and returns a Dict of lines read from the file
def read_files(file_location, train_type=bool):
    if train_type == True:
        read_data_train = {}
        read_data_dev = {}
        read_data_test = {}
        csv_file = open(file_location, encoding="UTF-8")
        read_csv = csv.reader(csv_file, delimiter = ",")

        len_data = len(pd.read_csv(file_location))
        split = len_data/3
        counter = 0
        for row in read_csv:
            #print(row[0])
            if row[0] == 'row_id':
                continue
            else:
                if counter <= split:
                    key = row[0]
                    read_data_train[key] = row[1:]
                    counter += 1
                elif split < counter <= split*2:
                    key = row[0]
                    read_data_dev[key] = row[1:]
                    counter += 1
                else:
                    key = row[0]
                    read_data_test[key] = row[1:]          

        csv_file.close() 
        return read_data_train, read_data_dev, read_data_test
    else:
        read_data_test = {}
        csv_file = open(file_location, encoding="UTF-8")
        read_csv = csv.reader(csv_file, delimiter = ",")
        for row in read_csv:
      
            if row[0] == 'row_id':
                continue
            else:
                key = row[0]
                read_data_test[key] = row[1:]
        csv_file.close() 
        return read_data_test



def process_data(data, data_out=bool):
    stop_words = set(stopwords.words('english'))
    if data_out == False:
        processed_data = {}
        for id,value in data.items():
            sent = nltk.sent_tokenize(value[0])

            '''del_index = (str(value[2]) + ' ' +str(value[3])).split()
            for x in range(len(del_index)):
                del_index[x] = int(del_index[x])'''

            for i in range(len(sent)):
                sent[i] = sent[i].lower()
                sent[i] = sent[i].strip()
                sent[i] = re.sub(r'\W', ' ', sent[i])
                sent[i] = re.sub(r'\s+', ' ', sent[i])

            word_list = sent[0].split(" ")
            filtered_words = [word for word in word_list if word not in stopwords.words('english') and word != '']
            value[0] = " ".join(filtered_words)
            
            class_type = value[1]
            if class_type not in processed_data.keys():
                processed_data[class_type] = str(value[0])
            else:
                processed_data[class_type] += ", " + value[0]

           
        return processed_data
    else:
        processed_data = {}
        for id,value in data.items():
            sent = nltk.sent_tokenize(value[0])

            del_index = (str(value[2]) + ' ' +str(value[3])).split()
            for x in range(len(del_index)):
                del_index[x] = int(del_index[x])

            for i in range(len(sent)):
                sent[i] = sent[i].lower()
                sent[i] = sent[i].strip()
                sent[i] = re.sub(r'\W', ' ', sent[i])
                sent[i] = re.sub(r'\s+', ' ', sent[i])

            word_list = sent[0].split(" ")
            filtered_words = [word for word in word_list if word not in stopwords.words('english') and word != '']
            value[0] = " ".join(filtered_words)
            
            row_id = id
            original_label = value[1]
            doc = value[0]
            processed_data[id] = [original_label, doc] 
        return processed_data        

        

def create_bag_of_words(data):
    # Creating the Bag of Words model
    bag_words = {}
    vocab = {}
    word_count = {}
    
    for id,value in data.items():
        processed_data = value
        
        words = nltk.word_tokenize(processed_data)
        for word in words:
            if word not in word_count.keys():
                word_count[word] = 1
            else:
                word_count[word] += 1

            if word not in vocab.keys():
                vocab[word] = 1
            else:
                vocab[word] += 1

        bag_words[id] = word_count
        word_count = {}   
    return bag_words, vocab


def output_file(output_location, data_out):

    # List containing the headers of our output file
    titles = ["original_label", "output_label","row_id"]

    # To open the output csv file in write mode
    with open(output_location, 'w', encoding='UTF8', newline='') as f:

        # Returns a writer object 
        writer = csv.writer(f)

        # write the header line to the file
        writer.writerow(titles)

        # For loop to iterate through all key, value pairs in the output data dict.
        for key,value in sorted(data_out.items()):
            row = []  # List to store the next row to be written

            
            # To append all the required details in the right order to the row list 

            for item in value:
                row.append(str(item))
            row.append(str(key)) 

            # To write the row to the output file
            writer.writerow(row)
            # To empty the row list to collect details for next row
            row = []

def dir_confusion(mat):
    #calculates confusion matrix for diretor
    sys_dir = [mat[0][0], mat[0][1]+mat[0][2]+mat[0][3]]
    sys_dir_not = [mat[1][0]+mat[2][0]+mat[3][0]]
    total = 0
    for i in mat:
        total += sum(i)
    not_true = total - sum(sys_dir)-sum(sys_dir_not)
    sys_dir_not.append(not_true)
    conf_dir = [sys_dir,sys_dir_not]
    numer = conf_dir[0][0]
    denom = sum(sys_dir)
    prec = numer/denom
    print('This is the confusion matrix for director: '+ str(conf_dir))
    return conf_dir,numer,denom,prec


def per_confusion(mat):
    #calculates confusion matrix for performer
    sys_per = [mat[1][1], mat[1][0]+mat[1][2]+mat[1][3]]
    sys_per_not = [mat[0][1]+mat[2][1]+mat[3][1]]
    total = 0
    for i in mat:
        total += sum(i)
    not_true = total - sum(sys_per)-sum(sys_per_not)
    sys_per_not.append(not_true)
    conf_per = [sys_per,sys_per_not]
    numer = conf_per[0][0]
    denom = sum(sys_per)
    prec = numer/denom
    print('This is the confusion matrix for performer: '+str(conf_per))
    return conf_per,numer,denom,prec

def cha_confusion(mat):
    #calculates confusion matrix for character
    sys_cha = [mat[2][2], mat[2][0]+mat[2][1]+mat[2][3]]
    sys_cha_not = [mat[0][2]+mat[1][2]+mat[3][2]]
    total = 0
    for i in mat:
        total += sum(i)
    not_true = total - sum(sys_cha)-sum(sys_cha_not)
    sys_cha_not.append(not_true)
    conf_cha = [sys_cha,sys_cha_not]
    numer = conf_cha[0][0]
    denom = sum(sys_cha)
    prec = numer/denom

    print('This is the confusion matrix for characters: '+str(conf_cha))
    return conf_cha,numer,denom,prec

def pub_confusion(mat):
    #calculates confusion matrix for publisher
    sys_pub = [mat[3][3], mat[3][0]+mat[3][1]+mat[3][2]]
    sys_pub_not = [mat[0][3]+mat[1][3]+mat[2][3]]
    total = 0
    for i in mat:
        total += sum(i)
    not_true = total - sum(sys_pub)-sum(sys_pub_not)
    sys_pub_not.append(not_true)
    conf_pub = [sys_pub,sys_pub_not]
    numer = conf_pub[0][0]
    denom = sum(sys_pub)
    prec = numer/denom

    print('This is the confusion matrix for publisher: '+ str(conf_pub))
    return conf_pub,numer,denom,prec
    

def confusion_matrix(data_out):
    dir = [0,0,0,0]
    per = [0,0,0,0]
    cha = [0,0,0,0]
    pub = [0,0,0,0]
    mat = []
    for key,value in data_out.items():
        if value[1] == 'director':
            if value[0] == 'director':
                dir[0]= dir[0]+1
            elif value[0] == 'performer':
                dir[1] = dir[1]+1
            elif value[0] == 'characters':
                dir[2] = dir[2]+1
            elif value[0] == 'publisher':
                dir[3] = dir[3]+1
        elif value[1] == 'performer':
            if value[0] == 'director':
                per[0] = per[0]+1
            elif value[0] == 'performer':
                per[1] = per[1]+1
            elif value[0] == 'characters':
                per[2] = per[2]+1
            elif value[0] == 'publisher':
                dir[3] = dir[3]+1
        elif value[1] == 'characters':
            if value[0] == 'director':
                cha[0] = cha[0]+1
            elif value[0] == 'performer':
                cha[1] = cha[1]+1
            elif value[0] == 'characters':
                cha[2] = cha[2]+1
            elif value[0] == 'publisher':
                cha[3] = cha[3]+1
        elif value[1] == 'publisher':
            if value[0] == 'director':
                pub[0] = pub[0]+1
            elif value[0] == 'performer':
                pub[1] = pub[1]+1
            elif value[0] == 'characters':
                pub[2] = pub[2]+1
            elif value[0] == 'publisher':
                pub[3] = pub[3]+1

    mat.append(dir)
    mat.append(per)
    mat.append(cha)
    mat.append(pub)
    
    print('This is the confusion matrix for the four classes: '+ str(mat))
    
    prec_dir =  dir[0]/sum(dir)
    prec_per = per[1]/sum(per)
    prec_cha = cha[2]/sum(cha)
    prec_pub = pub[3]/sum(pub)
    
    prec = [prec_dir,prec_per,prec_cha,prec_pub]
    ave = sum(prec)/4
    print('This is the average precison for the classes: ' + str(ave))
    print('This is the precison for each class. Read as[director,performer,character,publisher]: '+ str(prec))
    return mat,ave

def three_fold(input_train,input_dev,input_test):
    '''Function is used to train data and test data using the 3 different splits. it returns the values from the best model'''
    models = []
    max_accuracy = 0
    for i in range(3):
        if i == 0:
            set_1 = process_data(input_train, data_out=False)
            set_2 = process_data(input_dev, data_out=False)
            set_3 = process_data(input_test, data_out=True)
            bag_input_set1, set1_vocab = create_bag_of_words(set_1)
            bag_input_set2, set2_vocab = create_bag_of_words(set_2)
        elif i == 1:
            set_1 = process_data(input_test, data_out=False)
            set_2 = process_data(input_dev, data_out=False)
            set_3 = process_data(input_train, data_out=True)
            bag_input_set1, set1_vocab = create_bag_of_words(set_1)
            bag_input_set2, set2_vocab = create_bag_of_words(set_2)
        else:
            set_1 = process_data(input_test, data_out=False)
            set_2 = process_data(input_train, data_out=False)
            set_3 = process_data(input_dev, data_out=True)
            bag_input_set1, set1_vocab = create_bag_of_words(set_1)
            bag_input_set2, set2_vocab = create_bag_of_words(set_2)

        #set1 and 2 
        logprior_set12, log_likelihood_set12, v_set12,classes_set12 = naive_bayes_classifier(set_1|set_2,set1_vocab|set2_vocab,bag_input_set1|bag_input_set2)

        for key,item in set_3.items():
            
            value = test_naive_bayes(item[1],logprior_set12,log_likelihood_set12,bag_input_set1|bag_input_set2,v_set12,classes_set12)
            item[1] = value
        test_mat,ave = confusion_matrix(set_3)
        print('Average precision for fold',i,'is: '+str(ave))
        if ave > max_accuracy:
            max_accuracy = ave
            logprior, log_likelihood, v,classes = logprior_set12, log_likelihood_set12, v_set12,classes_set12
            bag_input = bag_input_set1|bag_input_set2
        

    return logprior, log_likelihood, v, bag_input, classes

def main():
    # To create an argument parser object
    parser = argparse.ArgumentParser()

    # To add arguments to the parser object
    parser.add_argument("--train", help="Please provide the location of the train data file")
    parser.add_argument("--test", help="Please provide the location of the test data file")
    parser.add_argument("--output", help="Please provide the location of the output file")
    parser.add_argument("-v", "--verbosity", help="Increase output verbosity", action= "store_true" )

    # parses arguments through the parse_args() method. This will inspect the command line, 
    # convert each argument to the appropriate type and then invoke the appropriate action.
    args = parser.parse_args()
    
    train = args.train  # Variable to store input directory path
    output = args.output  # Variable to store output file path 
    test = args.test  # Variable to store output file path 


    if args.verbosity:  # Helps the user check what input and output file paths they have provided
        print(f"the location of the data file is {train} ")
        print(f"the location of the cfg file is {test} ")
        print(f"the location of the output file is {output} ")


    input_train, input_dev, input_test = read_files(train, train_type=True)

    logprior_dev, log_likelihood_dev, v_dev,bag_input_dev, classes_dev = three_fold(input_train,input_dev,input_test)

    test_input= read_files(test, train_type=False)
    #processed_test = process_data(test_input, data_out=False)
    data_out = process_data(test_input, data_out=True)

    for key,item in data_out.items():
        value = test_naive_bayes(item[1],logprior_dev,log_likelihood_dev,bag_input_dev,v_dev,classes_dev)
        item[1] = value
   
    test_mat,ave = confusion_matrix(data_out)

    #obtain values from individual confusion matrix to calculate micro and macro avegrages
    conf_dir,numer_dir,denom_dir,prec_dir=dir_confusion(test_mat)
    conf_per,numer_perr,denom_per,prec_per=per_confusion(test_mat)
    conf_char,numer_char,denom_char,prec_char=cha_confusion(test_mat)
    conf_pub,numer_pub,denom_pub,prec_pub=pub_confusion(test_mat)

    # calulations of averages
    macro = (prec_dir+prec_per+prec_char+prec_pub)/4
    print('The macroaverage precison is '+ str(macro))

    micro = (numer_dir+numer_char+numer_perr+numer_pub)/(denom_dir+denom_char+denom_per+denom_pub)
    print('The microaverage precison is '+ str(micro))

    #wrtiting to output file
    output_file(output, data_out)


main()
    