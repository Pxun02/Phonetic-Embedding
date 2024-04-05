import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from util.params import Phoneme, All_phoneme
from util.onehotEncoder import OHEncoder
import itertools

def pad_df(df):
    max_length = df.astype(str).apply(lambda x: x.str.len()).max().max()
    df = df.astype(str).apply(lambda x: x.str.ljust(max_length, '0'))
    return df, max_length

def merge_str(arr1, arr2):
    res = []
    for s1 in arr1:
        for s2 in arr2:
            res.append(s1+s2)
    return res

def all_letter_lists(df, max_length):
    let_list = set()
    for column in df.columns:
        combined_values = ''.join(df[column].astype(str))
        unique_chars = set(combined_values)
        let_list.update(unique_chars)

    let_list = list(let_list)
    let_list = sorted(let_list, key=lambda x: ord(x))

    # print(let_list)
    # print(f'letter candidates: {len(let_list)}, word length: {max_length}')
    
    return let_list

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=100):
    train_loss_history = np.zeros(num_epochs)
    val_loss_history = np.zeros(num_epochs)

    # Train
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            inputs, train_labels = batch
            labels_initials = train_labels[:, 0].to(device)
            labels_finals = train_labels[:, 1].to(device)
            labels_tones = train_labels[:, 2].to(device)
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs_initials, outputs_finals, outputs_tones = model(inputs)
            loss_initials = criterion(outputs_initials, labels_initials)
            loss_finals = criterion(outputs_finals, labels_finals)
            loss_tones = criterion(outputs_tones, labels_tones)
            loss = loss_initials + loss_finals + loss_tones
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted_initials = torch.max(outputs_initials.data, 1)
            _, predicted_finals = torch.max(outputs_finals.data, 1)
            _, predicted_tones = torch.max(outputs_tones.data, 1)
            predicted = np.array([np.array(elem) for elem in zip(predicted_initials, predicted_finals, predicted_tones)])
            predicted = torch.tensor(predicted, dtype=torch.long)
            total += train_labels.size(0)

            train_cor = predicted.eq(train_labels)
            correct += np.array([all(inner_list) for inner_list in train_cor]).sum().item()
        
        train_loss_history[epoch] = total_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]: Loss = {total_loss / len(train_loader):.4f}, Accuracy = {correct / total:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs, val_labels = val_batch
                val_labels_initials = val_labels[:, 0].to(device)
                val_labels_finals = val_labels[:, 1].to(device)
                val_labels_tones = val_labels[:, 2].to(device)
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                
                val_outputs_initials, val_outputs_finals, val_outputs_tones = model(val_inputs)
                val_loss_initials = criterion(val_outputs_initials, val_labels_initials)
                val_loss_finals = criterion(val_outputs_finals, val_labels_finals)
                val_loss_tones = criterion(val_outputs_tones, val_labels_tones)
                val_loss += val_loss_initials + val_loss_finals + val_loss_tones
                _, val_predicted_initials = torch.max(val_outputs_initials.data, 1)
                _, val_predicted_finals = torch.max(val_outputs_finals.data, 1)
                _, val_predicted_tones = torch.max(val_outputs_tones.data, 1)

                val_predicted = np.array([np.array(elem) for elem in zip(val_predicted_initials, val_predicted_finals, val_predicted_tones)])
                val_predicted = torch.tensor(val_predicted, dtype=torch.long)
                val_total += val_labels.size(0)
                
                val_cor = val_predicted.eq(val_labels)
                val_correct += np.array([all(inner_list) for inner_list in val_cor]).sum().item()
        
            val_loss_history[epoch] = val_loss / len(val_loader)
        
        print(f"Validation: Loss = {val_loss / len(val_loader):.4f}, Accuracy = {val_correct / val_total:.4f}")

    return train_loss_history, val_loss_history

# plot loss history
def plot_loss_history(train_loss_history, val_loss_history, model_name, modes, save):
    plt.plot(train_loss_history, label='train_loss')
    plt.plot(val_loss_history, label='val_loss')

    plt.title('Loss history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid()

    if save:
        plt.savefig(f'./result/{model_name}/{model_name}_{modes}_loss_history')
    else:
        plt.show()
    
    plt.cla()

# test
def test(model, test_dataset, criterion, letter_list, max_length, input_langs, target_langs, no_print):
    test_decodeds = []
    test_decodeds2 = []
    test_preds = []
    test_actuals = []
    test_cor = []

    test_loss = 0.0

    test_cor_initials = 0
    test_cor_finals = 0
    test_cor_tones = 0

    test_correct = 0
    cand_correct = 0

    cand_cor_initials = 0
    cand_cor_finals = 0
    cand_cor_tones = 0

    phoneme = Phoneme(target_langs)

    for test_input, test_label in test_dataset:
        test_label_initials = test_label[0]
        test_label_finals = test_label[1]
        test_label_tones = test_label[2]

        test_outputs_initials, test_outputs_finals, test_outputs_tones = model(test_input)
        _, test_predicted_initials = torch.max(test_outputs_initials.data, 0)
        _, test_predicted_finals = torch.max(test_outputs_finals.data, 0)
        _, test_predicted_tones = torch.max(test_outputs_tones.data, 0)

        test_loss_initials = criterion(test_outputs_initials, test_label_initials)
        test_loss_finals = criterion(test_outputs_finals, test_label_finals)
        test_loss_tones = criterion(test_outputs_tones, test_label_tones)

        test_loss += test_loss_initials + test_loss_finals + test_loss_tones

        test_cor_initials += int((test_label_initials == test_predicted_initials))
        test_cor_finals += int((test_label_finals == test_predicted_finals))
        test_cor_tones += int((test_label_tones == test_predicted_tones))

        cor = int((test_label_initials == test_predicted_initials)) * int((test_label_finals == test_predicted_finals)) * int((test_label_tones == test_predicted_tones))
        test_cor.append('o' if cor else 'x')

        test_correct += cor

        # Decode Labels
        test_decoded = test_input[:len(test_input) // len(input_langs)].numpy().reshape(max_length, len(letter_list))
        test_decoded = ''.join([letter_list[id] for id in np.argmax(test_decoded, axis=1)]).rstrip('0')
        test_decodeds.append(test_decoded)

        if len(input_langs) > 1:
            test_decoded2 = test_input[len(test_input) // len(input_langs):].numpy().reshape(max_length, len(letter_list))
            test_decoded2 = ''.join([letter_list[id] for id in np.argmax(test_decoded2, axis=1)]).rstrip('0')
            test_decodeds2.append(test_decoded2)


        test_pred = ("" if test_predicted_initials == len(phoneme.initials) else phoneme.initials[test_predicted_initials]) + phoneme.finals[test_predicted_finals] + phoneme.tones[test_predicted_tones]
        test_actual = ("" if test_label_initials == len(phoneme.initials) else phoneme.initials[test_label_initials]) + phoneme.finals[test_label_finals] + phoneme.tones[test_label_tones]
        
        test_preds.append(test_pred)
        test_actuals.append(test_actual)

        # candidates list
        cand_predicted_initials = torch.topk(test_outputs_initials.data, 3).indices
        cand_predicted_finals = torch.topk(test_outputs_finals.data, 3).indices
        cand_predicted_tones = torch.topk(test_outputs_tones.data, 2).indices

        cand_cor_initials += int((test_label_initials in cand_predicted_initials))
        cand_cor_finals += int((test_label_finals in cand_predicted_finals))
        cand_cor_tones += int((test_label_tones in cand_predicted_tones))

        cor = int((test_label_initials in cand_predicted_initials)) * int((test_label_finals in cand_predicted_finals)) * int((test_label_tones in cand_predicted_tones))
        cand_correct += cor


    test_accuracies = [round(test_correct / len(test_dataset), 4), round(test_cor_initials / len(test_dataset), 4), round(test_cor_finals / len(test_dataset), 4), round(test_cor_tones / len(test_dataset), 4)]
    cand_accuracies = [round(cand_correct / len(test_dataset), 4), round(cand_cor_initials / len(test_dataset), 4), round(cand_cor_finals / len(test_dataset), 4), round(cand_cor_tones / len(test_dataset), 4)]

    if not no_print:
        print(f"Test: Loss = {test_loss / len(test_dataset):.4f}")
        print(f"Single Entry: \t Accuracy = {test_accuracies[0]}, Initial Acc: {test_accuracies[1]}, Finals Acc: {test_accuracies[2]}, Tones Acc: {test_accuracies[3]}")
        print(f"Candidates List: Accuracy = {cand_accuracies[0]}, Initial Acc: {cand_accuracies[1]}, Finals Acc: {cand_accuracies[2]}, Tones Acc: {cand_accuracies[3]}")

    if len(input_langs) > 1:
        return test_accuracies, cand_accuracies, pd.DataFrame(list(zip(test_cor, test_dataset.character, test_decodeds, test_decodeds2, test_preds, test_actuals)), columns=['correct?', 'Character', *input_langs, f'Predicted {target_langs}', f'Actual {target_langs}'])
    else:
        return test_accuracies, cand_accuracies, pd.DataFrame(list(zip(test_cor, test_dataset.character, test_decodeds, test_preds, test_actuals)), columns=['correct?', 'Character', *input_langs, f'Predicted {target_langs}', f'Actual {target_langs}'])


# mine rules
def minePhoneticShift(model, letter_list, max_length, input_langs, target_langs, flatten=True):
    phoneme = Phoneme(target_langs)
    initials_dict = dict(zip([*phoneme.initials, "∅"], [set() for i in range(len(phoneme.initials) + 1)]))
    finals_dict = dict(zip(phoneme.finals, [set() for i in range(len(phoneme.finals))]))
    tones_dict = dict(zip(phoneme.tones, [set() for i in range(len(phoneme.tones))]))
    for initial in [*phoneme.initials, "∅"]:
        for final in phoneme.finals:
            for tone in phoneme.tones:
                txt = initial+final+tone

                pad_phon = txt.ljust(max_length, '0')
                pad_phon = {input_langs[0]: [pad_phon]}
                pad_phon = pd.DataFrame(data=pad_phon)
                ip = OHEncoder(pad_phon, input_langs, letter_list)
                if flatten:
                    ip = ip.reshape(ip.shape[1] * ip.shape[2]).squeeze()

                ip = torch.from_numpy(ip)

                out_initials, out_finals, out_tones = model(ip)
                out_predicted_initials = torch.topk(out_initials.data, 1).indices
                for i in out_predicted_initials:
                    if i == len(phoneme.initials):
                        initials_dict[initial].add("∅")
                    else:
                        initials_dict[initial].add(phoneme.initials[i])

                out_predicted_finals = torch.topk(out_finals.data, 1).indices
                for i in out_predicted_finals:
                    finals_dict[final].add(phoneme.finals[i])

                out_predicted_tones = torch.topk(out_tones.data, 1).indices
                for i in out_predicted_tones:
                    tones_dict[tone].add(phoneme.tones[i])

    ans_initials = pd.DataFrame(list(zip([*phoneme.initials, "∅"], initials_dict.values())), columns=[input_langs[0], f'Predicted Initial {target_langs}'])   
    ans_finals = pd.DataFrame(list(zip(phoneme.finals, finals_dict.values())), columns=[input_langs[0], f'Predicted Final {target_langs}'])   
    ans_tones = pd.DataFrame(list(zip(phoneme.tones, tones_dict.values())), columns=[input_langs[0], f'Predicted Tone {target_langs}'])   
    
    print("Rules are generated!")
    return ans_initials, ans_finals, ans_tones

def mineRules_ver2(ans, input_lang, target_lang):
    input_pheneme = Phoneme(input_lang[-1])
    target_phoneme = Phoneme(target_lang)
    phoneme = dict(zip([input_lang[-1], target_lang], [input_pheneme, target_phoneme]))
    
    components = dict.fromkeys([input_lang[-1], target_lang], [])
    for lang in input_lang[-1], target_lang:
        comps = []
        for word in ans["Predicted " + lang if lang == target_lang else lang]:
            initial = None
            final = None
            tone = word[-1] if word[-1].isnumeric() else '∅'

            woTone = word[:-1] if word[-1].isnumeric() else word
            for i in range(len(woTone)):
                if woTone[i:] in phoneme[lang].finals:
                    final = woTone[i:]
                    initial = '∅' if i == 0 else woTone[:i]
                    break

            temp = [initial, final, tone]
            comps.append(temp)
        components[lang] = comps

    initials_dict = dict(zip([*phoneme[input_lang[-1]].initials, "∅"], [set() for _ in range(len(phoneme[input_lang[-1]].initials) + 1)]))
    finals_dict = dict(zip(phoneme[input_lang[-1]].finals, [set() for _ in range(len(phoneme[input_lang[-1]].finals))]))
    tones_dict = dict(zip([*phoneme[input_lang[-1]].tones, "∅"], [set() for _ in range(len(phoneme[input_lang[-1]].tones) + 1)]))
    for i in range(len(components[input_lang[-1]])):
        initials_dict[components[input_lang[-1]][i][0]].add(components[target_lang][i][0])
        finals_dict[components[input_lang[-1]][i][1]].add(components[target_lang][i][1])
        tones_dict[components[input_lang[-1]][i][2]].add(components[target_lang][i][2])

    ans_initials = pd.DataFrame(list(zip([*phoneme[input_lang[-1]].initials, "∅"], initials_dict.values())), columns=[input_lang[0], f'Predicted Initial {target_lang}'])   
    ans_finals = pd.DataFrame(list(zip(phoneme[input_lang[-1]].finals, finals_dict.values())), columns=[input_lang[0], f'Predicted Final {target_lang}'])   
    ans_tones = pd.DataFrame(list(zip([*phoneme[input_lang[-1]].tones, "∅"], tones_dict.values())), columns=[input_lang[0], f'Predicted Tone {target_lang}'])   

    print("Rules are generated!")
    return ans_initials, ans_finals, ans_tones

            
def performance_recorder(mode, model_name, total_accuracies, top3_accuracies):
    file_path = './result/performance.csv'
    df = None
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        pre_scheme = ['ltc_yue2cmn', 'yue2cmn', 'ltc_cmn2yue', 'cmn2yue']
        pre_model = ['MLP', 'CNN', 'RNN', 'LSTM', 'Transformer']
        combinations = list(itertools.product(pre_scheme, pre_model))
        result = [list(comb) for comb in combinations]
        padded_result = [arr + [0.0] * (10 - len(arr)) for arr in result]
        df = pd.DataFrame(padded_result, columns=['Scheme', 'Model', 'Accuracy', 'Initial Accuracy', 'Final Accuracy', 'Tone Accuracy', 'Top 3 Accuracy', 'Top 3 Initial Accuracy', 'Top 3 Final Accuracy', 'Top 3 Tone Accuracy'])


    cond = (df['Scheme']==mode) & (df['Model']==model_name)
    df.loc[cond, ['Accuracy', 'Initial Accuracy', 'Final Accuracy', 'Tone Accuracy']] = total_accuracies
    df.loc[cond, ['Top 3 Accuracy', 'Top 3 Initial Accuracy', 'Top 3 Final Accuracy', 'Top 3 Tone Accuracy']] = top3_accuracies
    df.to_csv('./result/performance.csv', index=False)
