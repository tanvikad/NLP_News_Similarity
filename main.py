import numpy as np
import argparse
from deep_translator import GoogleTranslator
import os
import json
import random
import math
from collections import Counter
#import semeval_8_2021_ia_downloader

import spacy

import en_core_web_sm
nlp = en_core_web_sm.load()

import matplotlib.pyplot as plt

#https://www.overleaf.com/7569536854wszndjmnxgsf
def test():
    print("Pipeline:", nlp.pipe_names)
    doc = nlp("I was reading the paper.")
    token = doc[0]  # 'I'
    print(token.morph)  # 'Case=Nom|Number=Sing|Person=1|PronType=Prs'
    print(token.morph.get("PronType"))  # ['Prs']

    translated = GoogleTranslator(source='auto', target='en').translate("Ameryka\u0144skie obligacje \u015bmieciowe (high yield) podro\u017ca\u0142y o 14%. Je\u015bli kto\u015b postawi\u0142 na Treasuries za po\u015brednictwem Vanguard Total Bond Market ETF, to po 2019 roku zainkasowa\u0142 8,7% zysku. Ca\u0142kiem nie\u017ale jak na \u201enudne\u201d i bezpieczne obligacje Wuja Sama.\n\nKompletny odlot zaliczy\u0142y suwerenne obligacje europejskie, gdzie szale\u0144stwo ujemnych st\u00f3p procentowych popycha\u0142o inwestor\u00f3w do spekulacji rodem z manii tulipanowej. ")  # output -> Weiter so, du bist großartig
    print(translated)


#"abcd_efgh = [abcd,efgh] - list of tuples"
def return_pairs_of_keys(file):
    """Reads from the csv file and retrieves the pairs of keys for articles"""
    pairs = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            values = str(line).split(",")
            #if(float(values[-3]) > 2): continue #currently this just skips all the false pairs in the csv
            keys = str(line).split(",")[2]
            keys = keys.split("_")
            if keys[0] == "pair":
                continue
            pairs +=[(values[0], values[1], int(keys[0]), int(keys[1]), float(values[-3]))]
    return pairs
             
def expand_pairs(pairs, false_rate):
    """Takes in existing (true) pairs and desired rate of false pairs.
    Generates and labels false pairs and combines randomly with the true pairs."""
    total_num = len(pairs)/(1-false_rate)
    #options = 2*len(pairs) - 1
    options = len(pairs) - 1
    return_pairs = []
    for pair in pairs:
        return_pairs += [(pair[0], pair[1], True)]
    while len(return_pairs) < total_num:
        first = random.randint(0, options)
        second = random.randint(0, options)
        if first == second:
            continue
    
        # first_elem = pairs[first // 2][first % 2]
        # second_elem = pairs[second // 2][second % 2]

        first_elem = pairs[first][0]
        second_elem = pairs[second][1]
        if (first_elem, second_elem) in pairs or (second_elem, first_elem) in pairs:
            continue
        if (first_elem, second_elem, False) in return_pairs or (second_elem, first_elem, False) in return_pairs:
            continue
        return_pairs += [(first_elem, second_elem, False)]
    
    random.shuffle(return_pairs)
    return return_pairs



#abcd --> json.text in folder [cd]
def get_text(key, multilingual=True):
    """Takes in the ID of the text. Returns the text"""
    last_two = str(key % 100) #get last two digits
    if(len(last_two) < 2):
        last_two = "0"+last_two #add preceding 0 to last_two if needed
    pathname = "Data/output_dir/" + last_two + "/" + str(key) + ".json"
    if(multilingual): pathname = "Data/output_dir_cross_and_mono/" + last_two + "/" + str(key) + ".json"
    try:
        f = open(pathname)
        data = json.load(f)
        #if(data["meta_lang"] != "en"): print("Different Language")
        return data['text']
    except:
        return None
    

def get_pos(text, pos_list):
    """Takes in text and list of desired parts of speech.
    Returns array containing words matching the desired POS"""
    arr = []
    doc = nlp(text)
    for token in doc:
        if (pos_list == None):
            arr += [str(token)]
        elif(token.pos_ in pos_list):
            arr += [str(token)]
    return arr

def classify(text1, text2, threshold = 0.31, pos_list=["PROPN"]):
    """Takes in two texts and a threshold for matching nouns/propn.
    Returns the frequency of matching nouns between the two texts,
    as well as true if the frequency is above the threshold"""
    pos_1 = set(get_pos(text1, pos_list))
    pos_2 = set(get_pos(text2, pos_list))
    #pos_1 = set(get_pos(text1, ["NOUN", "PROPN"]))
    #pos_2 = set(get_pos(text2, ["NOUN", "PROPN"]))
    #pos_1 = set(get_pos(text1, None))
    #pos_2 = set(get_pos(text2, None))
    count = 0
    for token in pos_1:
        if(token in pos_2):
            count += 1
    
    frequency = count / max(min(len(pos_1), len(pos_2)),1)
    return count, frequency, (frequency > threshold)


def translate_to_english(text):
    index = 0
    translated_text = ""

    while(index < len(text)):
        if(index + 1000 > len(text)):
            partial_translation = GoogleTranslator(source='auto', target='en').translate(text[index:])
            if(not partial_translation): break
            translated_text += partial_translation
            break
        partial_translation = GoogleTranslator(source='auto', target='en').translate(text[index:index+1000])
        if(not partial_translation): continue
        translated_text += partial_translation
        index += 1000
    
    return translated_text
def experiment(pairs_of_keys, translate = False):
    """We are currently not using this function."""
    i = 0
    train_split = 0.8
    train_test_index = int(100 * train_split) #int(len(pairs_of_keys)*train_split)
    training_pairs = pairs_of_keys[:train_test_index]
    test_pairs = pairs_of_keys[train_test_index:train_test_index+100]
    training_pairs = expand_pairs(training_pairs, 0.2)
    test_pairs = expand_pairs(test_pairs, 0.5)
    frequency_sum_true = 0
    frequency_sum_false = 0
    total_true = 0
    total_false = 0
    
    # For each pair in the training set, find the match rate between them
    for pair in training_pairs:
        i+=1
        input1, input2, groundtruth = pair
        text1 = get_text(input1)
        text2 = get_text(input2)
        #if translate is set to True, we translate both articles into english
        if translate:
            text1 = translate_to_english(text1)
            text2 = translate_to_english(text2)
        if(text1 == None or text2 == None): continue
        count, frequency, classfication = classify(text1, text2)
        if(groundtruth):
            frequency_sum_true += frequency
            total_true +=1
        else:
            frequency_sum_false += frequency
            total_false += 1
    
    #Gets the average match frequency of nouns/pronouns for true pairs vs. false pairs
    print("True average frequency", frequency_sum_true/total_true, "\n")
    print("False average frequency", frequency_sum_false/total_false, "\n")


    num_correct = 0
    num_total = len(test_pairs)
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    for pair in test_pairs:
        input1, input2, groundtruth = pair
        text1 = get_text(input1)
        text2 = get_text(input2)
        #if translate is set to True, we translate both articles into english
        if translate:
            text1 = GoogleTranslator(Fsource='auto', target='en').translate(text1)
            text2 = GoogleTranslator(source='auto', target='en').translate(text2)
        if(text1 == None or text2 == None): continue
        count, frequency, classfication = classify(text1, text2)
        if(classfication == groundtruth): 
            num_correct+=1
            if groundtruth:
                true_positive+=1
            else:
                true_negative+=1
        else:
            if groundtruth:
                false_negative+= 1
            else:
                false_positive += 1
                
    
    #Calculate f1 score and other metrics
    #precision: (true positive)/(true positive + false positive)
    precision = true_positive/(true_positive + false_positive)
    #recall: (true positive)/(true positive + false negative)
    recall = true_positive/(true_positive + false_negative)
    f1 = (2*precision*recall)/(precision + recall)



    print("Accuracy: ", num_correct/num_total)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("f1: ", f1)

    
def get_data(test_pairs, x, granularity=50, num_total=100, translate=False, use_correlation=False, random_data=False, pos_list=["PROPN"]):
    
    num_correct = []
    true_positive = []
    true_negative = []
    false_positive = []
    false_negative = []

    for i in range(granularity):
        num_correct += [0]
        true_positive += [0]
        true_negative += [0]
        false_positive += [0]
        false_negative += [0]

    for pair in test_pairs:
        la1, la2, input1, input2, groundtruth = pair
        groundtruth = groundtruth < 2.5
        text1 = get_text(input1)
        text2 = get_text(input2)
        if(text1 == None or text2 == None): continue
        if translate:
            text1 = translate_to_english(text1)
            text2 = translate_to_english(text2)
        count, frequency, classfication = classify(text1, text2, pos_list=pos_list)
        for i in range(granularity):
            trial_truth = frequency > x[i]
            if(random_data): (random.uniform(0,1) > x[i])
            if(trial_truth == groundtruth):
                num_correct[i] += 1   
                if groundtruth:
                    true_positive[i] += 1
                else:
                    true_negative[i] += 1
            else:
                if groundtruth:
                    false_negative[i] += 1
                else:
                    false_positive[i] += 1

               
    print(true_negative)
    print(true_positive)
    print(false_positive)
    print(false_negative)
    # #Calculate f1 score and other metrics

    #making num_correct into percentage
    print(num_correct)

    precisionarray = [0] * len(num_correct)
    recallarray = [0] * len(num_correct)
    f1array = [0] * len(num_correct)
    correlationarray = [0] * len(num_correct)
    for i in range(granularity):
        num_correct[i] = num_correct[i]/num_total
        if use_correlation:
            correlationnumerator = ((true_positive[i] * true_negative[i]) - (false_positive[i]*false_negative[i]))
            correlationdenominator = math.sqrt((true_positive[i]+false_positive[i])*(true_positive[i]+false_negative[i])*(true_negative[i]+false_positive[i])*(true_negative[i]+false_negative[i]))
            correlationarray[i] = correlationnumerator/correlationdenominator
        #Since we had a problem where everything was predicted false, use if statement for precision
        if true_positive[i] == 0 and false_positive[i] == 0:
            print("predicted all false")
            precisionarray[i] = 0
        else:
            precisionarray[i] = true_positive[i]/(true_positive[i] + false_positive[i])
        # set recall
        if true_positive[i] + false_negative[i] == 0:
            recallarray[i] = 0
        else:
            recallarray[i] = true_positive[i]/(true_positive[i] + false_negative[i])
        #set f1 to zero if recall and precision are both 0
        if precisionarray[i] == 0 and recallarray[i] == 0:
            f1array[i] = 0
        else:
            f1array[i] = (2*precisionarray[i]*recallarray[i])/(precisionarray[i] + recallarray[i])
    
    #We have all the f1 scores for each threshold index. Get the best one and its index
    bestf1value = max(f1array)
    bestf1index = f1array.index(bestf1value)
    #Find the precision and recall corresponding to the best f1 score
    bestf1precision = precisionarray[bestf1index]
    bestf1recall = recallarray[bestf1index]

    print("The best f1 is ", bestf1value)
    print("The best precision is ", bestf1precision, "\n")
    print("The best recall is ", bestf1recall, "\n")
    if use_correlation:
        return correlationarray, bestf1precision, bestf1recall
    else:
        return  f1array, bestf1precision, bestf1recall


def get_path_of_image(title):
    path_name = ""
    for char in title:
        if char == ' ':
            path_name += '_'
        else:
            path_name += char
    path_name += ".jpg"
    return path_name



def using_different_POS(pairs_of_keys, num_total = 100, granularity = 20, min = 0.0, max = 1.0):
    random.shuffle(pairs_of_keys)
    test_pairs = pairs_of_keys[:num_total]
    #test_pairs = expand_pairs(test_pairs, 0.5) #we don't use expand_pairs anymore
    num_total = len(test_pairs)

    x = []
    step_size = (max-min)/granularity
    for i in range(granularity):
        x += [min+(i*step_size)]

    print(x)
    propn_data, a1, a2 = get_data(test_pairs,x, granularity=granularity, num_total=num_total, translate=False, random_data=False, pos_list=["PROPN"])
    all_nouns_data, a1, a2 = get_data(test_pairs,x, granularity=granularity, num_total=num_total, translate=False, random_data=False, pos_list=["PROPN", "NOUN"])
    propn_num_data, a1, a2 = get_data(test_pairs,x, granularity=granularity, num_total=num_total, translate=False, random_data=False, pos_list=["PROPN", "NUM"])
    all_data, a1, a2 = get_data(test_pairs,x, granularity=granularity, num_total=num_total, translate=False, random_data=False, pos_list=None)

    
    #pos_2 = set(get_pos(text2, ["NOUN", "PROPN"]))

    plt.plot(x,propn_data, marker=".", label="only Proper Nouns", color="Blue")
    plt.plot(x,all_nouns_data, marker=".", label="all Nouns", color="Red")
    plt.plot(x,propn_num_data, marker=".", label="Proper Nouns and Numbers", color="Orange")
    plt.plot(x,all_data, marker=".", label="all words", color="Green")
    plt.legend()
    title = "Exploring use of different POS"
    plt.title(title)
    plt.xlabel("Binary Classification")
    plt.ylabel("F1")
    plt.savefig(get_path_of_image(title))
    


def get_best_classifier(pairs_of_keys, num_total = 100, granularity = 20, min = 0.0, max = 1.0, plot_f1 = True):
    random.shuffle(pairs_of_keys)
    test_pairs = pairs_of_keys[:num_total]
    #test_pairs = expand_pairs(test_pairs, 0.5) #we don't use expand_pairs anymore
    num_total = len(test_pairs)

    x = []
    step_size = (max-min)/granularity
    for i in range(granularity):
        x += [min+(i*step_size)]

    print(x)
    translated_data, bestf1precisiontranslated, bestf1recalltranslated = get_data(test_pairs,x, granularity=granularity, num_total=num_total, translate=True, random_data=False)
    not_translated_data, bestf1precisionnottranslated, bestf1recallnottranslated = get_data(test_pairs,x, granularity=granularity, num_total=num_total, translate=False, random_data=False)



    plt.plot(x,translated_data, marker=".", label="Translated Data", color="Blue")
    plt.plot(x,not_translated_data, marker=".", label="Not Translated Data", color="Red")
    plt.legend()
    title = "Binary Classifier vs. F1 with Proper Nouns"
    plt.title(title)
    plt.xlabel("Binary Classification")
    plt.ylabel("F1")
    plt.savefig(get_path_of_image(title))

    if(plot_f1):
        plot_f_scores(bestf1precisiontranslated, bestf1recalltranslated, bestf1precisionnottranslated, bestf1recallnottranslated)
    


    
def plot_f_scores(precision, recall, precision_nottranslated, recall_nottranslated):
    fig, ax = plt.subplots() 
    fig.clear(True) 
    f1 = (2*precision*recall)/(precision + recall)
    fs = [f1]
    fn1 = (2*precision_nottranslated*recall_nottranslated)/(precision_nottranslated + recall_nottranslated)
    fns = [fn1]
    logBeta = [0]
    for i in range(1, 10):
        scale = 1.2**i
        invScale = 1.0/scale
        f = ((1 + scale**2)*precision*recall)/(precision*scale**2 + recall)
        fn = ((1 + scale**2)*precision_nottranslated*recall_nottranslated)/(precision_nottranslated*scale**2 + recall_nottranslated)
        fs += [f]
        fns += [fn]
        logBeta += [math.log(scale)]
        f2 = ((1 + invScale**2)*precision*recall)/(precision*invScale**2 + recall)
        fn2 = ((1 + invScale**2)*precision_nottranslated*recall_nottranslated)/(precision_nottranslated*invScale**2 + recall_nottranslated)
        fs = [f2] + fs
        fns = [fn2] + fns
        logBeta = [math.log(invScale)] + logBeta

        
    
    plt.plot(logBeta,fs, marker=".", label="Translated", color="Blue")
    plt.plot(logBeta,fns, marker=".", label="Not Translated", color="Red")
    plt.legend()
    title = "F scores vs log(Beta) with Proper Nouns"
    plt.title(title)
    plt.xlabel("log(Beta)")
    plt.ylabel("F Scores")
    plt.savefig(get_path_of_image(title))


#should only be with monolingual pairs 
def finding_difference_in_language(pairs_of_keys, num_total = 100, translate=False, granularity = 20, min = 0.0, max = 1.0, plot_f1 = True):
    random.shuffle(pairs_of_keys)
    train_split = 0.7
    train_pairs = pairs_of_keys[:int(0.7*len(pairs_of_keys))]
    
    dict = {}
    for i in range(len(train_pairs)):
        if(train_pairs[i][0] not in dict):
            dict[train_pairs[i][0]] = [train_pairs[i]]
        else:
            dict[train_pairs[i][0]] += [train_pairs[i]]
    

    for key in dict.keys():
        print(key)
        print(len(dict[key]))

    x = []
    step_size = (max-min)/granularity
    for i in range(granularity):
        x += [min+(i*step_size)]

    print(x)

    i =0
    colors = ["Blue", "Red", "Orange", "Green", "Purple", "Yellow", "lime", "slategray"]
    language_code_dict = {"en":"English", "de":"German", "tr":"Turkish", "pl":"Polish", "es":"Spanish", "ar":"Arabic", "fr":"French"}
    for key in dict.keys():
        print(language_code_dict[key])
        data, data_precision, data_recall = get_data(dict[key],x, granularity=granularity, num_total=len(dict[key]), translate=translate, random_data=False)
        plt.plot(x,data, marker=".", label=language_code_dict[key], color=colors[i])
        print(i)
        i += 1
    
    plt.legend()
    title = "Translated: Exploring Classification on different monolingual pairs"
    plt.title(title)
    plt.xlabel("Binary Classification")
    plt.ylabel("F1")
    plt.savefig(get_path_of_image(title))
    

def main(args):
    pairs_of_keys = return_pairs_of_keys(args.file)
    #get_best_classifier(pairs_of_keys)
    finding_difference_in_language(pairs_of_keys,translate=True)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='SemEval Task 7')
    parser.add_argument("--file", "-f", metavar="FILE", required=True,
                        help="Read csv data from directory")
    args = parser.parse_args()
    main(args)
