import numpy as np

def OHEncoder(df, input_labels, let_list):
    def one_hot_encode(sequence, letter_list):
        encoding = np.zeros((len(sequence), len(letter_list)))
        for i, aa in enumerate(sequence):
            if aa in letter_list:
                index = letter_list.index(aa)
                encoding[i, index] = 1
        return encoding

    temp = []

    for column in input_labels:
        ser = df[column]
        temptemp = []
        for sequence in ser:
            temptemp.append(one_hot_encode(sequence, let_list))
        temptemp = np.array(temptemp, dtype=np.float32)
        temp.append(temptemp)
    
    if len(temp) > 1:
        temp = np.concatenate(temp, axis=1)
    else:
        temp = temp[0]
        
    return temp

def labels_combination(df, target_column, initials, finals, tones):
    labels = []
    for col in df[target_column]:
        # initials
        initial = 0
        initial_len = 0
        for i, ini in enumerate(initials):
            if col[0:2] == ini:
                initial = i
                initial_len = 2
                break
            elif col[0] == ini:
                initial = i
                initial_len = 1
                break
        if initial_len == 0:
            initial = len(initials)
        
        # finals
        final = 0
        for i, fin in enumerate(finals):
            if col[initial_len:-1] == fin:
                final = i
                break
        
        # tones
        tone = 0
        for i, tn in enumerate(tones):
            if col[-1] == tn:
                tone = i
                break
        labels.append(np.array([initial, final, tone]))
    labels = np.array(labels)
    return labels