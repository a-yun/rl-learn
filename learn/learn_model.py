import numpy as np
import sys
import pickle
from random import shuffle, seed, randint
import math
from data import Data
import os
from utils_learn import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb

N_ACTIONS = 18
GLOVE_DIM = 50
VOCAB_SIZE = 296
INFERSENT_DIM = 4096

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if self.args.action_enc == 'frequency':
            self.action_enc_linear1 = nn.Linear(N_ACTIONS, args.action_enc_size)
        elif self.args.action_enc == 'rnn':
            self.action_enc_emb = nn.Embedding(N_ACTIONS, args.action_enc_size)
            nn.init.xavier_uniform(self.action_enc_emb.weight)
            self.action_rnn = nn.LSTM(args.action_enc_size, args.action_enc_size, 1, batch_first=True)
            self.action_enc_linear1 = nn.Linear(args.action_enc_size, args.action_enc_size)
        else:
            raise NotImplementedError

        self.action_enc_linear2 = nn.Linear(args.action_enc_size, args.action_enc_size)
        self.action_enc_linear3 = nn.Linear(args.action_enc_size, args.action_enc_size)
        self.action_enc_bn1 = nn.BatchNorm1d(args.action_enc_size)
        self.action_enc_bn2 = nn.BatchNorm1d(args.action_enc_size)
        self.action_enc_bn3 = nn.BatchNorm1d(args.action_enc_size)

        if args.lang_enc == 'infersent':
            self.lang_enc_linear = nn.Linear(INFERSENT_DIM, args.lang_enc_size)
        elif args.lang_enc == 'onehot':
            self.lang_enc_emb = nn.Embedding(VOCAB_SIZE, GLOVE_DIM)
            nn.init.xavier_uniform(self.lang_enc_emb.weight)
            self.lang_enc_lstm = nn.LSTM(
                GLOVE_DIM, 
                args.lang_enc_size, 
                num_layers=1, 
                batch_first=True)
        elif args.lang_enc == 'glove':
            self.lang_enc_lstm = nn.LSTM(
                GLOVE_DIM, 
                args.lang_enc_size, 
                num_layers=1, 
                batch_first=True)
        else:
            raise NotImplementedError

        self.classifier_linear1 = nn.Linear(
            args.action_enc_size + args.lang_enc_size, args.classifier_size)
        self.classifier_linear2 = nn.Linear(args.classifier_size, args.classifier_size)
        self.classifier_linear3 = nn.Linear(args.classifier_size, 2)
        self.classifier_bn1 = nn.BatchNorm1d(args.classifier_size)
        self.classifier_bn2 = nn.BatchNorm1d(args.classifier_size)

    def forward(self, action_vector, lang, lengths, action_lengths=None):
        if self.args.action_enc == 'frequency':
            action_enc = F.relu(self.action_enc_bn1(self.action_enc_linear1(action_vector)))
            action_enc = F.relu(self.action_enc_bn2(self.action_enc_linear2(action_enc)))
            action_enc = self.action_enc_bn3(self.action_enc_linear3(action_enc))
        elif self.args.action_enc == 'rnn':
            action_enc = self.action_enc_emb(action_vector)

            action_enc = nn.utils.rnn.pack_padded_sequence(action_enc, action_lengths, batch_first=True, enforce_sorted=False)
            action_enc, (h_n, c_n) = self.action_rnn(action_enc)
            action_enc = nn.utils.rnn.pad_packed_sequence(action_enc, batch_first=True)[0]
            seq_selector = (action_lengths - 1).unsqueeze(1).unsqueeze(1).expand(-1, 1, action_enc.shape[2])
            action_enc = torch.gather(action_enc, 1, seq_selector).squeeze(1)

            action_enc = self.action_enc_bn3(self.action_enc_linear3(action_enc))
        else:
            raise NotImplementedError

        if self.args.lang_enc == 'infersent':
            lang_encoded = self.lang_enc_linear(lang)
        elif self.args.lang_enc == 'onehot':
            lang = self.lang_enc_emb(lang)
            lang = nn.Dropout(p=0.2)(lang)

            lang = nn.utils.rnn.pack_padded_sequence(lang, lengths, batch_first=True)
            output, (h_n, c_n) = self.lang_enc_lstm(lang)
            output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
            output = torch.sum(output, dim=1)
            lang_encoded = output / torch.unsqueeze(lengths, 1).float().cuda()
            lang_encoded = nn.Dropout(p=0.2)(lang_encoded)
        elif self.args.lang_enc == 'glove':
            lang = nn.Dropout(p=0.2)(lang)
            lang = nn.utils.rnn.pack_padded_sequence(lang, lengths, batch_first=True)
            output, (h_n, c_n) = self.lang_enc_lstm(lang)
            output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
            output = torch.sum(output, dim=1)
            lang_encoded = output / torch.unsqueeze(lengths, 1).float().cuda()
            lang_encoded = nn.Dropout(p=0.2)(lang_encoded)

        action_lang = torch.cat((action_enc, lang_encoded), dim=1)
        output = F.relu(self.classifier_bn1(self.classifier_linear1(action_lang)))
        output = F.relu(self.classifier_bn2(self.classifier_linear2(output)))
        output = self.classifier_linear3(output)

        return output


class LearnModel(object):
    def __init__(self, mode, args=None, model_dir=None):
        if mode == 'train':
            self.args = args
            self.data = Data(args)
            self.model = Model(args).cuda()
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.args.lr, 
                weight_decay=self.args.weight_decay)
        elif mode == 'predict':
            ckpt = torch.load(model_dir)
            self.args = ckpt['args']
            self.model = Model(self.args).cuda()
            self.model.load_state_dict(ckpt['state_dict'])
            self.model.eval()

    def pad_seq_feature(self, seq, length):
            seq = np.asarray(seq)
            if length < np.size(seq, 0):
                    return seq[:length]
            dim = np.size(seq, 1)
            result = np.zeros((length, dim))
            result[0:seq.shape[0], :] = seq
            return result

    def pad_seq_onehot(self, seq, length):
            seq = np.asarray(seq)
            if length < np.size(seq, 0):
                return seq[:length]
            result = np.zeros(length)
            result[0:seq.shape[0]] = seq
            return result

    def get_batch_lang_lengths(self, lang_list):
        if self.args.lang_enc == 'onehot':
            lengths = np.array([len(l) for l in lang_list])
            langs = np.array([self.pad_seq_onehot(l, max(lengths)) for l in lang_list])
            return langs, lengths
        elif self.args.lang_enc == 'glove':
            lengths = np.array([len(l) for l in lang_list])
            langs = np.array([self.pad_seq_feature(l, max(lengths)) for l in lang_list])
            return langs, lengths
        elif self.args.lang_enc == 'infersent':
            return np.array(lang_list), np.array([])
        else:
            raise NotImplementedError

    def get_batch_action_lengths(self, action_list):
        lengths = np.array([len(l) for l in action_list])
        actions = np.array([self.pad_seq_onehot(l, max(lengths)) for l in action_list])
        return actions, lengths

    def run_batch(self, data, start, is_train):
        if is_train:
            self.model.train()
            self.optimizer.zero_grad()

        # lang = self.args.lang_enc
        curr_batch_data = data[start:start+self.args.batch_size]

        action_length_list = None
        action_list, lang_list, label_list = zip(*curr_batch_data)
        lang_list = np.array(lang_list)
        lang_list, length_list = self.get_batch_lang_lengths(lang_list)

        if self.args.action_enc == 'frequency':
            action_list = torch.Tensor(action_list).float().cuda()
        elif self.args.action_enc == 'rnn':
            action_list = list(action_list)
            action_list, action_length_list = self.get_batch_action_lengths(action_list)

            action_list = torch.from_numpy(action_list).long().cuda()
            action_length_list = torch.from_numpy(action_length_list).long().cuda()

        label_list = torch.Tensor(label_list).long().cuda()
        if self.args.lang_enc == 'onehot':
            lang_list = torch.from_numpy(lang_list).long().cuda()
            length_list = torch.from_numpy(length_list).long().cuda()
        else:
            lang_list = torch.from_numpy(lang_list).float().cuda()
            length_list = torch.from_numpy(length_list).long().cuda()

        if self.args.lang_enc == 'onehot' or self.args.lang_enc == 'glove':
            indices = torch.sort(length_list, descending=True)[1]
            action_list = action_list[indices]
            lang_list = lang_list[indices]
            length_list = length_list[indices]
            label_list = label_list[indices]
            if self.args.action_enc == 'rnn':
                action_length_list = action_length_list[indices]

        pred = self.model(action_list, lang_list, length_list, action_length_list)
        loss = torch.nn.CrossEntropyLoss()(pred, label_list)

        if is_train:
            loss.backward()
            self.optimizer.step()

        return loss.detach() * len(label_list), torch.argmax(pred, dim=-1), label_list

    def run_epoch(self, data, is_train):
        start = 0
        loss = 0
        labels = []
        pred = []

        while start < len(data):
            batch_loss, batch_pred, batch_labels = self.run_batch(data, start, is_train)

            start += self.args.batch_size
            loss += batch_loss    
            pred += list(batch_pred)
            labels += list(batch_labels)

        correct = np.sum([1.0 if x == y else 0.0 for (x, y) in zip(pred, labels)])
        return correct / len(data), loss / len(data)

    def train_network(self):
        steps_per_epoch = int(math.ceil(len(self.data.train_data) / self.args.batch_size))
        n_epochs = 50

        epoch_start = 0
        best_val_acc = 0
        for epoch in range(epoch_start, n_epochs):
            shuffle(self.data.train_data)
            acc_train, loss_train = self.run_epoch(self.data.train_data, is_train=1)
            acc_valid, loss_valid = self.run_epoch(self.data.valid_data, is_train=0)

            print('Epoch: %d \t TL: %f \t VL: %f \t TA: %f \t VA: %f' %
                (epoch, loss_train, loss_valid, acc_train, acc_valid))

            if acc_valid > best_val_acc:
                if self.args.model_file:
                    state = {
                        'args': self.args,
                        'epoch': epoch,
                        'best_val_acc': best_val_acc,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    }
                    torch.save(state, self.args.model_file)

    def predict(self, action_list, lang_list):
        action_length_list = None
        if self.args.action_enc == 'frequency':
            s = np.sum(action_list)
            action_list = np.array(action_list)
            if s > 0:
                action_list /= s
            lang_list, length_list = self.get_batch_lang_lengths(lang_list)
            action_list = torch.Tensor(action_list).float().cuda()
        elif self.args.action_enc == 'rnn':
            action_list = np.array(action_list)
            lang_list, length_list = self.get_batch_lang_lengths(lang_list)

            action_list = list(action_list)
            action_list, action_length_list = self.get_batch_action_lengths(action_list)
            action_list = torch.from_numpy(action_list).long().cuda()
            action_length_list = torch.from_numpy(action_length_list).long().cuda()

        if self.args.lang_enc == 'onehot':
            lang_list = torch.from_numpy(lang_list).long().cuda()
            length_list = torch.from_numpy(length_list).long().cuda()
        else:
            lang_list = torch.from_numpy(lang_list).float().cuda()
            length_list = torch.from_numpy(length_list).long().cuda()

        pred = self.model(action_list, lang_list, length_list, action_length_list)[0]
        return pred.data.cpu().numpy()

