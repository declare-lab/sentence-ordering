import csv, json, pandas as pd

def get_filenames(split):
    with open(split, "r") as inp:
        filenames = inp.read()
    filenames = filenames.split('\n')[:-1]
    return filenames

def get_story_text(data):
    story_sentences = {}
    annotations = data['annotations']
    for annotation in annotations:
        story_id = annotation[0]['story_id']
        story_sentences.setdefault(story_id, [])
        story_sentences[story_id].append(annotation[0]['original_text'])
    return story_sentences

# For NIPS, AAN, NSF
def write_data(task, split, split_name):
    directory = 'data/' + task + '/'
    dpath = directory + 'split/' + split
    files = get_filenames(dpath)
    
    outname = directory +  split_name + '.tsv'

    with open(outname, "w") as out:
        tsv_writer = csv.writer(out, delimiter='\t')

        for file in files:
            if task == 'nips':
                with open(directory + 'txt_tokenized/' + 'a' + file + '.txt', 'r') as inp:
                    lines = inp.readlines()
            else:
                with open(directory + 'txt_tokenized/' + file, 'r') as inp:
                    lines = inp.readlines()

            lines = [line.strip() for line in lines]
            tsv_writer.writerow(lines)


# For SIND
def write_data_sind(split):
    data = json.load(open('data/sind/' + split + '.story-in-sequence.json','r'))
    story_sentences = get_story_text(data)
    
    if split == 'val':
        split_name = 'valid'
    else:
        split_name = split
        
    outname = 'data/sind/' +  split_name + '.tsv'
    
    with open(outname, "w") as out:
        tsv_writer = csv.writer(out, delimiter='\t')
        for story_id in story_sentences.keys():
            story = story_sentences[story_id]
            tsv_writer.writerow(story)


# For ROC
def write_data_roc(split):
    df = pd.read_csv('data/roc/' + split + '.csv')
    outname = 'data/roc/' + split + '.tsv'
    
    with open(outname, "w") as out:
        tsv_writer = csv.writer(out, delimiter='\t')
        for i in range(len(df)):
            row = df.iloc[i]
            story = [row['sentence'+str(j)] for j in range(1, 6)]
            tsv_writer.writerow(story)


if __name__ == "__main__":
    write_data('nips', '2013le_papers', 'train')
    write_data('nips', '2014_papers', 'valid')
    write_data('nips', '2015_papers', 'test')

    for task in ['nsf', 'aan']:
        write_data(task, 'train', 'train')
        write_data(task, 'valid', 'valid')
        write_data(task, 'test', 'test')
        
    write_data_sind('train')
    write_data_sind('val')
    write_data_sind('test')

    write_data_roc('train')
    write_data_roc('valid')
    write_data_roc('test')
