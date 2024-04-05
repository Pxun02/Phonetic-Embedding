import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torchinfo import summary
from torch import nn, optim
import torch

from util.params import Phoneme, Mode, MLP_params, CNN_params, RNN_params, LSTM_params, Transformer_params
from util.onehotEncoder import OHEncoder, labels_combination
from util.utility import pad_df, all_letter_lists, train, plot_loss_history, test, minePhoneticShift, mineRules_ver2, performance_recorder
from util.dataloader import create_dataset, create_dataloader
from models.myModels import MLP, CNN, RNN, LSTM, Transformer


# mode: 'ltc_yue2cmn', 'yue2cmn', 'ltc_cmn2yue', 'cmn2yue'

def main(df2, df3):
    model_names = ['MLP', 'CNN', 'RNN', 'LSTM', 'Transformer']
    modes = ['ltc_yue2cmn', 'yue2cmn', 'ltc_cmn2yue', 'cmn2yue']

    print('\n*** Master Runner ***')
    for model_name in model_names:
        for mode in modes:
            print(f'\n*** Model: {model_name}, mode: {mode} ***')
            if mode == 'ltc_yue2cmn' or mode == 'ltc_cmn2yue':
                main_model(df3, model_name, mode, save_fig=True, no_print=True)
            else:
                main_model(df2, model_name, mode, save_fig=True, no_print=True)


def main_model(df, model_name, mode, save_fig, no_print):
    # create the processed dataframe
    df2 = df.drop(columns=['Character']).copy()
    df2, max_length = pad_df(df2)
    
    # Character lists
    cc = list(df['Character'])

    # set target variables per mode
    mode_params = Mode(mode)
    input_labels, target_label = mode_params.input_labels, mode_params.target_label

    phoneme = Phoneme(target_label)
    # One-hot Encoding
    let_list = all_letter_lists(df2, max_length)
    inputs = OHEncoder(df2, input_labels, let_list)
    # Generate one-hot encoded target label
    label = labels_combination(df, target_label, phoneme.initials, phoneme.finals, phoneme.tones)

    # create dataset
    flatten = True if (model_name == 'MLP' or model_name == 'CNN') else False
    train_dataset, val_dataset, test_dataset = create_dataset(inputs, label, cc, flatten=flatten)

    # create dataloader
    train_loader, val_loader = create_dataloader(train_dataset, val_dataset)

    # Compose Model
    model_params = None
    model = None
    match model_name:
        case 'MLP':
            model_params = MLP_params(train_dataset, target_label)
            model = MLP(*model_params.getParams())
        case 'CNN':
            model_params = CNN_params(train_dataset, target_label, let_list)
            model = CNN(*model_params.getParams())
        case 'RNN':
            model_params = RNN_params(train_dataset, target_label)
            model = RNN(*model_params.getParams())
        case 'LSTM':
            model_params = LSTM_params(train_dataset, target_label)
            model = LSTM(*model_params.getParams())
        case 'Transformer':
            model_params = Transformer_params(train_dataset, target_label, max_length)
            model = Transformer(*model_params.getParams())

    
    # Display model summary
    if not no_print:
        print(summary(model, (model_params.getSummaryParams())))

    # selections
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_params.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    train_loss_history, val_loss_history = train(model, 
                                                 train_loader, 
                                                 val_loader, 
                                                 optimizer, 
                                                 criterion, 
                                                 device, 
                                                 num_epochs=model_params.num_epochs)
    
    # Plot loss history
    plot_loss_history(train_loss_history, val_loss_history, model_name, mode, save_fig)

    # test
    test_accuracies, cand_accuracies, ans = test(model, test_dataset, criterion, let_list, max_length, input_labels, target_label, no_print)

    # save
    low_mod = model_name.lower()
    ans.to_csv(f'./result/{low_mod}/{low_mod}_{mode}.csv', index=False)
    performance_recorder(mode, model_name, test_accuracies, cand_accuracies)

    # mine_rules
    # rule_initial, rule_final, rule_tone = minePhoneticShift(model, let_list, max_length, input_labels, target_label, flatten)
    rule_initial, rule_final, rule_tone = mineRules_ver2(ans, input_labels, target_label)
    rule_initial.to_csv(f'./result/{low_mod}/rules/rule_initial_{low_mod}_{mode}.csv', index=False)
    rule_final.to_csv(f'./result/{low_mod}/rules/rule_final_{low_mod}_{mode}.csv', index=False)
    rule_tone.to_csv(f'./result/{low_mod}/rules/rule_tone_{low_mod}_{mode}.csv', index=False)