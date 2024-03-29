import pickle
import json
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
import random

#concept2id, id2concept = concept2id, id2concept = load_resources()
with open('./data/cpnet/concept.txt', "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]

def load_resources(cpnet_vocab_path):
    merged_relations = [
    'antonym',
    'atlocation',
    'capableof',
    'causes',
    'createdby',
    'isa',
    'desires',
    'hassubevent',
    'partof',
    'hascontext',
    'hasproperty',
    'madeof',
    'notcapableof',
    'notdesires',
    'receivesaction',
    'relatedto',
    'usedfor',
    ]
    concept2id = {}
    relation2id = {}
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}
    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}
    return concept2id, id2concept

def get_pos(triple):
    subj = triple[0]
    rel = triple[1]
    obj = triple[2]
    
    
    with open('POS.pickle', 'rb') as handle: 
        PartofSpeech = pickle.load(handle)
    part = {}
    code2Pos = {'CC' : 'other',
'CD' : 'other',
'DT' : 'other',
'EX' : 'other',
'FW' : 'other',
'IN' : 'other',
'JJ' : 'adjective', 
'JJR' : 'adjective',
'JJS' : 'adjective',
'LS' : 'other', 
'MD' : 'other',
'NN' : 'noun',
'NNS' : 'noun', 
'NNP' : 'noun',
'NNPS' : 'noun',
'PDT' : 'other',
'POS' : 'other',
'PRP' : 'pronoun',
'PRP$' : 'pronoun',
'RB' : 'adverb',
'RBR' : 'adverb',
'RBS' : 'adverb',
'RP' : 'other',
'TO' : 'other',
'UH' : 'other',
'VB' : 'verb',
'VBD' : 'verb',
'VBG' : 'verb',
'VBN' : 'verb',
'VBP' : 'verb',
'VBZ' : 'verb',
'WDT' : 'other',
'WP' : 'other',
'WP$' : 'other',
'WRB' : 'other',
'not_found' : 'not_found'}
    try:
        t = PartofSpeech[subj]
    except KeyError:
        t = 'not_found'
    try:
        s = PartofSpeech[obj]
    except KeyError:
        s = 'not_found'
    if 'ing' in subj:
        t = 'VB'
    if 'ing' in obj:
        s = 'VB'
    if t in list(code2Pos.keys()):
        part['subject'] = code2Pos[t]
    else:
        part['subject'] = 'not_found'
    if s in list(code2Pos.keys()):
        part['object'] = code2Pos[s]
    else:
        part['object'] = 'not_found'
    return part
    
def get_negative(triple):
    part_of_speech = get_pos(triple)
    subj = triple[0]
    rel = triple[1]
    obj = triple[2]
    global id2concept
    #id2concept = triple[3]
    changed_subject = 'organic_compound'
    changed_object = 'most_important_meal_of_day'
    neg_triples = []
    relation_transform = {'antonym': ['isa', 'partof','hascontext','hasproperty','relatedto'],
    'atlocation': ['notcapableof', 'createdby', 'madeof', 'usedfor', 'causes'],
    'capableof' : ['createdby', 'madeof', 'notcapableof'],
    'causes' : ['atlocation', 'createdby', 'madeof',],
    'createdby': ['antonym', 'atlocation', 'causes', 'usedfor'],
    'isa' : ['antonym', 'usedfor', 'receivesaction'],
    'desires' : ['notdesires', 'antonym', 'notcapableof'],
    'hassubevent': ['causes', 'createdby'],
    'partof': ['antonym', 'notdesires', 'notcapableof'],
    'hascontext': ['antonym', 'notdesires', 'notcapableof'],
    'hasproperty': ['antonym', 'notdesires', 'notcapableof', 'receivesaction'],
    'madeof': ['antonym', 'atlocation', 'receivesaction'],
    'notcapableof': ['atlocation', 'capableof', 'createdby', 'hascontext'],
    'notdesires': ['atlocation', 'desires', 'receivesaction', 'createdby'],
    'receivesaction': ['atlocation', 'createdby', 'hascontext', 'hasproperty'],
    'relatedto': ['antonym', 'notcapableof', 'notdesires'],
    'usedfor': ['atlocation', 'notcapableof', 'notdesires']
    }
    
    #if (part_of_speech['subject'] == 'noun' or part_of_speech['object'] == 'noun') and rel!='antonym':
    #    neg_triples.append([subj, 'antonym', obj])
    #if (part_of_speech['object'] == 'noun' or part_of_speech['object'] == 'verb') and rel!='hasproperty':
    #    neg_triples.append([subj, 'hasproperty', obj])
    #if part_of_speech['object'] != 'noun' and rel!='atlocation':
    #    neg_triples.append([subj, 'atlocation', obj])
    #if part_of_speech['object'] == 'adjective' and rel!='createdby':
    #    neg_triples.append([subj, 'createdby', obj])
    #if part_of_speech['subject'] == 'adjective' and rel!='usedfor':
    #    neg_triples.append([subj, 'usedfor', obj])
    #if (part_of_speech['subject'] != part_of_speech['object']) and rel!='relatedto':
    #    neg_triples.append([subj,'relatedto',obj])
    #if subj != changed_subject and rel!='relatedto':
    #    neg_triples.append([changed_subject, 'relatedto', obj])
    #if obj != changed_object and rel!='relatedto':
    #    neg_triples.append([subj, 'relatedto', changed_object])
    #if rel!='antonym' and rel!='relatedto':
    #    neg_triples.append([obj, rel, subj])
    #for relation in relation_transform[rel]:
    #    if [subj, relation, obj] not in neg_triples:
    #        neg_triples.append([subj, relation, obj])
    r = random.choice(range(17))
    while list(relation_transform.keys())[r] == rel:
       r = random.choice(range(17)) 
    t1 = random.choice(range(799273))
    while id2concept[t1] == subj:
        t1 = random.choice(range(799273))
    t2 = random.choice(range(799273))
    while id2concept[t2] == obj:
        t2 = random.choice(range(799273))
    neg_triples.append([subj,list(relation_transform.keys())[r],obj])  
    neg_triples.append([id2concept[t1], rel, obj])
    neg_triples.append([subj, rel, id2concept[t2]])
    return neg_triples

def getAllneg(cpnet_csv_path, output_csv_path,id2concept):
    nrow = sum(1 for _ in open(cpnet_csv_path, 'r', encoding='utf-8'))
    triples = []
    with open(cpnet_csv_path, "r", encoding="utf8") as fin:
        for line in tqdm(fin, total=nrow):
            ls = line.strip().split('\t')
            rel = ls[0]
            subj = ls[1]
            obj = ls[2]
            weight = float(ls[3])
            #wrong_triples = get_negative(subj, rel, obj)
            #for triple in wrong_triples:
            triples.append([subj, rel, obj])
    with open(output_csv_path, "w", encoding="utf8") as fout:
        with Pool(cpu_count()) as p:
            for wrong_triple in tqdm(p.imap(get_negative,triples),total = len(triples)):
                for fact in wrong_triple:
                    fout.write('\t'.join(fact) + '\n')
        

def main():
    
    getAllneg('./data/cpnet/conceptnet.en.csv', 'neg_triples.csv', id2concept)
    
if __name__ == '__main__':
    main()