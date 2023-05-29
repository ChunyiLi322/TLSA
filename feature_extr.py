import codecs
import csv
# from stanza.nlp.corenlp import CoreNLPClient
import stanza
from stanza.server import CoreNLPClient
from stanfordcorenlp import StanfordCoreNLP
import logging


'''other function'''

def stanford_nlp_parse(nlp, sentence_data):
	
    '''English word segmentation'''
    log_record = open("log_stanford_nlp"+ path_to_file +"_new.txt", mode = "a+", encoding = "utf-8")
    print('-------------------------------Tokenize:', nlp.word_tokenize(sentence_data),file = log_record)
    #print('-------------------------------Tokenize:', nlp.word_tokenize(sentence_data)[0], type(nlp.word_tokenize(sentence_data)[0])) 
    print('-------------------------------Part of Speech:', nlp.pos_tag(sentence_data),file = log_record)
    #print('-------------------------------Named Entities:', nlp.ner(sentence_data),file = log_record)
    #print('-------------------------------Constituency Parsing:', nlp.parse(sentence_data),file = log_record)
    #print('-------------------------------Dependency Parsing:', nlp.dependency_parse(sentence_data),file = log_record)
    #print('-------------------------------Dependency Parsing:', nlp.dependency_parse(sentence_data)[0][0],type(nlp.dependency_parse(sentence_data)[0][0]))
    
    
    '''yi cun guan xi ti qu'''
    den_count = 0
    for den_name in nlp.dependency_parse(sentence_data):
        if den_name[0] == "neg" or den_name[0] == "cc" or den_name[0] == "mark" or den_name[0] == "aux":
           den_count = den_count +1
    print('-------------------------------den_count:',den_count)
    
    
    '''shu ju rong yu'''
    print('-------------------------------len(nlp.word_tokenize(sentence_data)):',len(nlp.word_tokenize(sentence_data)),file = log_record)
    print('-------------------------------len(nlp.pos_tag(sentence_data)):',len(nlp.pos_tag(sentence_data)) ,file = log_record)
     
    
    tokenize_list = nlp.word_tokenize(sentence_data)
    delete_list = []
    for pos_name,i in zip(nlp.pos_tag(sentence_data),range(0,len(nlp.pos_tag(sentence_data)))):
	    if pos_name[1] == "DT":
	       #print('-------------------------------i:',i)
	       delete_list.append(i)
    delete_list = sorted(delete_list, reverse=True)
    print('-------------------------------delete_list:',delete_list ,file = log_record)
    for delete_line in delete_list:
        tokenize_list.pop(delete_line) 
	    
	
    #print('-------------------------------tokenize_list:',tokenize_list)  
    tokenize_list.insert(len(tokenize_list)-1,str(den_count))
    #print('-------------------------------tokenize_list:',' '.join(tokenize_list))
    log_record.close()  
    return ' '.join(tokenize_list)  

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')
def preproc_sentence(_s):
    _s = unicode_to_ascii(_s.lower().strip())
    _s = re.sub(r"(['.-])", r" \1 ", _s)
    _s = re.sub(r'[" "]+', " ", _s)
    _s = re.sub(r"['][^.a-zA-Z-] ", " ", _s)
    _s = _s.strip()
    _s = '<start> ' + _s + ' <end>'
    return _s


'''
eng_sentence = u" May I borrow this book? "
fra_sentence = u"Puis-je emprunter ce livre?"
eng = preproc_sentence(eng_sentence)
fra = preproc_sentence(fra_sentence)

#  May I borrow this book?   ->   <start> may i borrow this book ? <end> 
#  Puis-je emprunter ce livre?  ->  <start> puis - je emprunter ce livre ? <end>  
'''

def create_dataset(_path_to_file, num_examples):
    lines = open(_path_to_file, encoding='UTF-8').read().strip().split('\n')
    random.shuffle(lines)
    word_pairs = [[preproc_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    print("word_pairs的长度 ",len(word_pairs))
    # print("word_pairs的真实值 ",word_pairs)
    word_pairs1 = [[preproc_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    #random.shuffle(word_pairs)
    random.shuffle(word_pairs)
    for i in range(0,10):
        test_datelist.append(word_pairs.pop(-1))
    print("word_pairs鐨勭被鍨 ",type(word_pairs))
    print("鍘熸湁鏁版嵁闆嗛暱搴 ",len(word_pairs1))
    print("鍒犲噺鏁版嵁闆嗛暱搴 ",len(word_pairs))
    return zip(*word_pairs1)



def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')  # padding = post (1,2,3,4,5,0,0,0,0,0)

    return tensor, lang_tokenizer


'''preprocessing '''
path_to_file = 'always.csv.dev'


stanza.install_corenlp()
nlp = StanfordCoreNLP(r'/home/lcy/adsample/NCR2Code/',lang='en',quiet=False,logging_level=logging.DEBUG)
file_list = []
file_line_list = []


'''read data'''
with open(path_to_file, 'r') as f:
    for line in f:
        line = line.split('\t') 
        file_line_list.append(stanford_nlp_parse(nlp,line[0]))
        file_line_list.append(line[1])
        file_list.append(file_line_list)
        file_line_list = []
	    
print('-------------------------------file_list:', file_list)

'''write data'''
with open(path_to_file,'w')as file:
     for line in file_list:
         file.write(line[0])
         file.write('\t')
         file.write(line[1])

