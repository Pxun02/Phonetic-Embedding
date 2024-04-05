import pandas as pd

class Phoneme():
    def __init__(self, target):
        self.initials = None
        self.finals = None
        self.tones = None
        if target == 'Mandarin':
            self.initials = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's']
            self.finals = ['e', 'a', 'ei', 'ai', 'ou', 'ao', 'en', 'an', 'eng', 'ang', 'er', 
                    'ya', 'yi', 'ye', 'you', 'yao', 'yin', 'yan', 'ying', 'yang', 
                    'i', 'ie', 'ia', 'iu', 'iao', 'in', 'ian', 'ing', 'iang', 
                    'wu', 'wo', 'wa', 'wei', 'wai', 'wen', 'wan', 'weng', 'wang', 
                    'u', 'uo', 'ue', 'o', 'ua', 'ui', 'uai', 'un', 'uan', 'ong', 'uang',
                    'yu', 'yue', 'yun', 'yuan', 'yong', 
                    'ü', 'üe', 'ün', 'üan', 'iong']
            self.tones = ['1', '2', '3', '4']
        
        elif target == 'Cantonese':
            self.initials = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'ng', 'h', 'gw', 'kw', 'w', 'z', 'c', 's', 'j']
            self.finals = ['aa', 'aai', 'aau', 'aam', 'aan', 'aang', 'aap', 'aat', 'aak',
                           'a', 'ai', 'au', 'am', 'an', 'ang', 'ap', 'at', 'ak',
                           'e', 'ei', 'eu', 'em', 'eng', 'ep', 'ek',
                           'i', 'iu', 'im', 'in', 'ing', 'ip', 'it', 'ik',
                           'o', 'oi', 'ou', 'on', 'ong', 'ot', 'ok',
                           'u', 'ui', 'un', 'ung', 'ut', 'uk',
                           'eoi', 'eon', 'eot', 'oe', 'oeng', 'oet', 'oek',
                           'yu', 'yun', 'yut', 'm', 'ng']
            self.tones = ['1', '2', '3', '4', '5', '6']

class All_phoneme():
    def __init__(self, input_langs):
        self.phon = None
        if 'Mandarin' in input_langs:
            self.phon = pd.read_csv('./dataset/mandarin_all_phoneme.csv')

class Mode():
    def __init__(self, mode):
        self.input_labels = None
        self.target_label = None
        match mode:
            case 'ltc_yue2cmn':
                self.input_labels = ['Middle Chinese', 'Cantonese']
                self.target_label = 'Mandarin'
            case 'yue2cmn':
                self.input_labels = ['Cantonese']
                self.target_label = 'Mandarin'
            case 'ltc_cmn2yue':
                self.input_labels = ['Middle Chinese', 'Mandarin']
                self.target_label = 'Cantonese'
            case 'cmn2yue':
                self.input_labels = ['Mandarin']
                self.target_label = 'Cantonese'

class MLP_params():
    def __init__(self, train_dataset, target):
        self.batch_size = 128
        self.input_size = train_dataset[0][0].shape[0]
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.learning_rate = 0.001
        self.num_epochs = 100
        phoneme = Phoneme(target)
        self.num_classes_initials = len(phoneme.initials) + 1
        self.num_classes_finals = len(phoneme.finals)
        self.num_classes_tones = len(phoneme.tones)
    
    def getParams(self):
        return self.input_size, self.hidden_size1, self.hidden_size2, self.num_classes_initials, self.num_classes_finals, self.num_classes_tones

    def getSummaryParams(self):
        return self.batch_size, self.input_size
    
class CNN_params():
    def __init__(self, train_dataset, target, let_list):
        self.batch_size = 128
        self.input_size = train_dataset[0][0].shape[0]
        self.input_channel = 1
        self.hidden_size1 = 5
        self.hidden_size2 = 256
        self.hidden_size3 = 64
        self.learning_rate = 0.005
        self.num_epochs = 50
        self.kernel_size = len(let_list)
        phoneme = Phoneme(target)
        self.num_classes_initials = len(phoneme.initials) + 1
        self.num_classes_finals = len(phoneme.finals)
        self.num_classes_tones = len(phoneme.tones)
    
    def getParams(self):
        return self.input_channel, self.input_size, self.hidden_size1, self.hidden_size2, self.hidden_size3, self.num_classes_initials, self.num_classes_finals, self.num_classes_tones, self.kernel_size

    def getSummaryParams(self):
        return self.input_channel, self.input_size
    
class RNN_params():
    def __init__(self, train_dataset, target):
        self.batch_size = 128
        self.input_size = train_dataset[0][0].shape
        self.num_layers = 3
        self.hidden_size = 256
        self.learning_rate = 0.0002
        self.num_epochs = 50
        phoneme = Phoneme(target)
        self.num_classes_initials = len(phoneme.initials) + 1
        self.num_classes_finals = len(phoneme.finals)
        self.num_classes_tones = len(phoneme.tones)

    def getParams(self):
        return self.input_size, self.hidden_size, self.num_layers, self.num_classes_initials, self.num_classes_finals, self.num_classes_tones

    def getSummaryParams(self):
        return self.input_size

class LSTM_params():
    def __init__(self, train_dataset, target):
        self.batch_size = 128
        self.input_size = train_dataset[0][0].shape
        self.num_layers = 3
        self.hidden_size = 256
        self.learning_rate = 0.0002
        self.num_epochs = 50
        phoneme = Phoneme(target)
        self.num_classes_initials = len(phoneme.initials) + 1
        self.num_classes_finals = len(phoneme.finals)
        self.num_classes_tones = len(phoneme.tones)

    def getParams(self):
        return self.input_size, self.hidden_size, self.num_layers, self.num_classes_initials, self.num_classes_finals, self.num_classes_tones

    def getSummaryParams(self):
        return self.input_size

class Transformer_params():
    def __init__(self, train_dataset, target, max_length):
        self.batch_size = 128
        self.input_size = train_dataset[0][0].shape
        self.num_layers = 3
        self.hidden_size = 128
        self.learning_rate = 0.0002
        self.num_epochs = 50
        phoneme = Phoneme(target)
        self.num_classes_initials = len(phoneme.initials) + 1
        self.num_classes_finals = len(phoneme.finals)
        self.num_classes_tones = len(phoneme.tones)
        self.max_length = max_length    

    def getParams(self):
        return self.input_size, self.hidden_size, self.num_layers, self.num_classes_initials, self.num_classes_finals, self.num_classes_tones, self.max_length
    
    def getSummaryParams(self):
        return self.input_size