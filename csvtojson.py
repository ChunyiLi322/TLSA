import json

id = 3930316

with open("sometimes.csv.train",'r')as file:
     for line in file:
        line_dict = {}
        labels_index_list = []
        line = line.split('\t') 
        line_dict["testid"] = str(id)
        line_dict["features_content"] = line[0].replace(',', '').replace('.', '').split(' ')
        labels_index_list.append(line[1].replace('\n', ''))
        line_dict["labels_index"] = labels_index_list
        line_dict["labels_num"] = 1
        line_json = json.dumps(line_dict)
        f2 = open('Train_sample.json', 'a')
        f2.write(line_json)
        f2.write('\n')
        f2.close()
        id = id + 1

        
with open("yuhan.csv.train",'r')as file:
     for line in file:
        line_dict = {}
        labels_index_list = []
        line = line.split('\t') 
        line_dict["testid"] = str(id)
        line_dict["features_content"] = line[0].replace(',', '').replace('.', '').split(' ')
        labels_index_list.append(line[1].replace('\n', ''))
        line_dict["labels_index"] = labels_index_list
        line_dict["labels_num"] = 1
        line_json = json.dumps(line_dict)
        f2 = open('Train_sample.json', 'a')
        f2.write(line_json)
        f2.write('\n')
        f2.close()
        id = id + 1
        
        
        
with open("next.csv.train",'r')as file:
     for line in file:
        line_dict = {}
        labels_index_list = []
        line = line.split('\t') 
        line_dict["testid"] = str(id)
        line_dict["features_content"] = line[0].replace(',', '').replace('.', '').split(' ')
        labels_index_list.append(line[1].replace('\n', ''))
        line_dict["labels_index"] = labels_index_list
        line_dict["labels_num"] = 1
        line_json = json.dumps(line_dict)
        f2 = open('Train_sample.json', 'a')
        f2.write(line_json)
        f2.write('\n')
        f2.close()
        id = id + 1
        
        
with open("always.csv.train",'r')as file:
     for line in file:
        line_dict = {}
        labels_index_list = []
        line = line.split('\t') 
        line_dict["testid"] = str(id)
        line_dict["features_content"] = line[0].replace(',', '').replace('.', '').split(' ')
        labels_index_list.append(line[1].replace('\n', ''))
        line_dict["labels_index"] = labels_index_list
        line_dict["labels_num"] = 1
        line_json = json.dumps(line_dict)
        f2 = open('Train_sample.json', 'a')
        f2.write(line_json)
        f2.write('\n')
        f2.close()
        id = id + 1        
        