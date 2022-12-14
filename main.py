import numpy as np
import argparse
from deep_translator import GoogleTranslator
import os
import json
import random
from collections import Counter
#import semeval_8_2021_ia_downloader

import spacy

import en_core_web_sm
nlp = en_core_web_sm.load()

import matplotlib.pyplot as plt


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
    pairs = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            keys = str(line).split(",")[2]
            keys = keys.split("_")
            if keys[0] == "pair":
                continue
            pairs +=[(int(keys[0]), int(keys[1]))]
    return pairs
             
def expand_pairs(pairs, false_rate):
    """Takes in existing (true) pairs and desired rate of false pairs.
    Generates and labels false pairs and combines randomly with the true pairs."""
    total_num = len(pairs)/(1-false_rate)
    options = 2*len(pairs) - 1
    return_pairs = []
    for pair in pairs:
        return_pairs += [(pair[0], pair[1], True)]
    while len(return_pairs) < total_num:
        first = random.randint(0, options)
        second = random.randint(0, options)
        if first == second:
            continue
    
        first_elem = pairs[first // 2][first % 2]
        second_elem = pairs[second // 2][second % 2]
        if (first_elem, second_elem) in pairs or (second_elem, first_elem) in pairs:
            continue
        if (first_elem, second_elem, False) in return_pairs or (second_elem, first_elem, False) in return_pairs:
            continue
        return_pairs += [(first_elem, second_elem, False)]
    
    random.shuffle(return_pairs)
    return return_pairs



#abcd --> json.text in folder [cd]
def get_text(key):
    """Takes in the ID of the text. Returns the text"""
    last_two = str(key % 100) #get last two digits
    if(len(last_two) < 2):
        last_two = "0"+last_two #add preceding 0 to last_two if needed
    pathname = "Data/output_dir/" + last_two + "/" + str(key) + ".json"
    try:
        f = open(pathname)
        data = json.load(f)
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

def classify(text1, text2, threshold = 0.31):
    """Takes in two texts and a threshold for matching nouns/propn.
    Returns the frequency of matching nouns between the two texts,
    as well as true if the frequency is above the threshold"""
    pos_1 = set(get_pos(text1, ["NOUN", "PROPN"]))
    pos_2 = set(get_pos(text2, ["NOUN", "PROPN"]))
    #pos_1 = set(get_pos(text1, ["PROPN"]))
    #pos_2 = set(get_pos(text2, ["PROPN"]))
    count = 0
    for token in pos_1:
        if(token in pos_2):
            count += 1
    
    frequency = count / max(min(len(pos_1), len(pos_2)),1)
    return count, frequency, (frequency > threshold)


def experiment(pairs_of_keys):
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
    for pair in training_pairs:
        i+=1
        input1, input2, groundtruth = pair
        text1 = get_text(input1)
        text2 = get_text(input2)
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
    for pair in test_pairs:
        input1, input2, groundtruth = pair
        text1 = get_text(input1)
        text2 = get_text(input2)
        if(text1 == None or text2 == None): continue
        count, frequency, classfication = classify(text1, text2)
        if(classfication == groundtruth): num_correct+=1
    
    print("We did this well: ", num_correct/num_total)
        



def get_best_classifier(pairs_of_keys, num_total = 100, granularity = 50, min = 0.0, max = 0.5):
    test_pairs = pairs_of_keys[:num_total]
    test_pairs = expand_pairs(test_pairs, 0.5)
    num_total = len(test_pairs)

    x = []
    num_correct = []
    
    step_size = (max-min)/granularity
    for i in range(granularity):
        x += [min+(i*step_size)]
        num_correct += [0]
    
    print(x)

    for pair in test_pairs:
        input1, input2, groundtruth = pair
        text1 = get_text(input1)
        text2 = get_text(input2)
        if(text1 == None or text2 == None): continue
        count, frequency, classfication = classify(text1, text2)
        for i in range(granularity):
            trial_truth = frequency > x[i]
            if(trial_truth == groundtruth):
                num_correct[i] += 1   


    #making num_correct into percentage
    print(num_correct)
    for i in range(granularity):
        num_correct[i] = num_correct[i]/num_total
    
    plt.plot(x,num_correct, marker=".")
    plt.title("Binary Classifier versus Accuracy with all Nouns")
    plt.xlabel("Binary Classification")
    plt.ylabel("Accuracy")
    plt.savefig("Binary_Classifier.jpg")
        
def main(args):
    pairs_of_keys = return_pairs_of_keys(args.file)
    get_best_classifier(pairs_of_keys)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='SemEval Task 7')
    parser.add_argument("--file", "-f", metavar="FILE", required=True,
                        help="Read csv data from directory")
    args = parser.parse_args()
    main(args)
