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

def classify(text1, text2, threshold = 0.31):
    """Takes in two texts and a threshold for matching nouns/propn.
    Returns the frequency of matching nouns between the two texts,
    as well as true if the frequency is above the threshold"""
    #pos_1 = set(get_pos(text1, ["NOUN", "PROPN"]))
    #pos_2 = set(get_pos(text2, ["NOUN", "PROPN"]))
    pos_1 = set(get_pos(text1, None))
    pos_2 = set(get_pos(text2, None))
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
            translated_text += partial_translation
            break
        partial_translation = GoogleTranslator(source='auto', target='en').translate(text[index:index+1000])
        if(not partial_translation): continue
        translated_text += partial_translation
        index += 1000
    
    return translated_text
def experiment(pairs_of_keys, translate = False):
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

    

def get_data(test_pairs, x, granularity=50, num_total=100, translate=True, use_F1=True):
    
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
        input1, input2, groundtruth = pair
        text1 = get_text(input1)
        text2 = get_text(input2)
        if(text1 == None or text2 == None): continue
        if translate:
            text1 = translate_to_english(text1)
            text2 = translate_to_english(text2)
        count, frequency, classfication = classify(text1, text2)
        for i in range(granularity):
            trial_truth = frequency > x[i]
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
    for i in range(granularity):
        num_correct[i] = num_correct[i]/num_total

        #Since we had a problem where everything was predicted false, use if statement for precision
        if true_positive[i] == 0 and false_positive[i] == 0:
            print("predicted all false")
            precisionarray[i] = 0
        else:
            precisionarray[i] = true_positive[i]/(true_positive[i] + false_positive[i])
        # set recall
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

    print(len(f1array))
    print(len(x))
    return  f1array, bestf1precision, bestf1recall



def get_best_classifier(pairs_of_keys, num_total = 100, granularity = 50, min = 0.0, max = 0.5, plot_f1 = True):
    random.shuffle(pairs_of_keys)
    test_pairs = pairs_of_keys[:num_total]
    test_pairs = expand_pairs(test_pairs, 0.5)
    num_total = len(test_pairs)

    x = []
    step_size = (max-min)/granularity
    for i in range(granularity):
        x += [min+(i*step_size)]

    print(x)
    translated_data, bestf1precisiontranslated, bestf1recalltranslated= get_data(test_pairs,x, granularity=granularity, num_total=num_total, translate=True)
    not_translated_data, bestf1precisionnottranslated, bestf1recallnottranslated= get_data(test_pairs,x, granularity=granularity, num_total=num_total, translate=False)
    
    
    plt.plot(x,translated_data, marker=".", label="Translated Data", color="Blue")
    plt.plot(x,not_translated_data, marker=".", label="Not Translated Data", color="Red")
    plt.legend()
    plt.title("Binary Classifier vs. F1 with all words")
    plt.xlabel("Binary Classification")
    plt.ylabel("F1")
    plt.savefig("Binary_Classifier.jpg")

    if(plot_f1):
        plot_f_scores(bestf1precisiontranslated, bestf1recalltranslated, bestf1precisionnottranslated, bestf1recallnottranslated)
    


    
def plot_f_scores(precision, recall, precision_nottranslated, recall_nottranslated):
    fig, ax = plt.subplots() 
    fig.clear(True) 
    f1 = (2*precision*recall)/(precision + recall)
    fs = [f1]
    logBeta_translated = [0]
    for i in range(1, 10):
        scale = 1.2**i
        invScale = 1.0/scale
        f = ((1 + scale**2)*precision*recall)/(precision*scale**2 + recall)
        fs += [f]
        logBeta_translated += [math.log(scale)]
        f2 = ((1 + invScale**2)*precision*recall)/(precision*invScale**2 + recall)
        fs = [f2] + fs
        logBeta_translated = [math.log(invScale)] + logBeta_translated
    
    logBeta_not_translated = [0]
    for i in range(1, 10):
        scale = 1.2**i
        invScale = 1.0/scale
        f = ((1 + scale**2)*precision*recall)/(precision*scale**2 + recall)
        fs += [f]
        logBeta_not_translated += [math.log(scale)]
        f2 = ((1 + invScale**2)*precision*recall)/(precision*invScale**2 + recall)
        fs = [f2] + fs
        logBeta_not_translated = [math.log(invScale)] + logBeta_not_translated
        
    
    plt.plot(logBeta_translated,fs, marker=".", label="Translated", color="Blue")
    plt.plot(logBeta_not_translated,fs, marker=".", label="Not Translated", color="Red")
    plt.legend()
    plt.title("F scores across many different Beta values graphed as log(Beta)")
    plt.xlabel("log(Beta)")
    plt.ylabel("F Scores")
    plt.savefig("F_scores.jpg")

    
def main(args):
    pairs_of_keys = return_pairs_of_keys(args.file)
    get_best_classifier(pairs_of_keys)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='SemEval Task 7')
    parser.add_argument("--file", "-f", metavar="FILE", required=True,
                        help="Read csv data from directory")
    args = parser.parse_args()
    main(args)
