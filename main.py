#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (c) 2023 Iñaki Amatria-Barral
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import tempfile
import subprocess
import progressbar

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from itertools import product

from argparse import ArgumentParser

from lpi_prediction import LPIModel
from lpi_prediction import LPIPredictor
from lpi_prediction import LPIDataEncoder
from lpi_prediction import LPIDataProvider
from lpi_prediction import LPILabelsProvider

from sklearn.model_selection import StratifiedKFold

class MIRLOModel(LPIModel):
    def _build_model(self, params):
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.BatchNormalization())
        for _ in range(params['layers']):
            model.add(
                tf.keras.layers.Dense(activation='elu', units=params['neurons'])
            )
            model.add(tf.keras.layers.Dropout(params['dropout']))
        model.add(tf.keras.layers.Dense(2, activation='softmax'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=params['learning_rate']
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _tune_hyperparameters(self, X, y):
        hyperparams = {
            'epochs': [50],
            'layers': [1],
            'neurons': [32],
            'dropout': [.5],
            'learning_rate': [.01]
        }

        params = []
        for param_values in product(*hyperparams.values()):
            params.append(dict(zip(hyperparams.keys(), param_values)))

        accuracies = [0] * len(params)
        skf = StratifiedKFold(n_splits=5, shuffle=True)

        bar = progressbar.ProgressBar(
            min_value=0,
            max_value=len(params) * 5,
            widgets=[
                'Hyperparameter tuning... ',
                '(', progressbar.AbsoluteETA(), ')'
            ]
        ).start()
        if len(params) > 1:
            for train_index, test_index in skf.split(X, y[:, 1]):
                X_train, X_val = X[train_index], X[test_index]
                y_train, y_val = y[train_index], y[test_index]
                for i, param in enumerate(params):
                    for _ in range(5):
                        model = self._build_model(param)
                        history = model.fit(
                            X_train,
                            y_train,
                            verbose=0,
                            epochs=param['epochs'],
                            validation_data=(X_val, y_val)
                        )
                        accuracies[i] += history.history['val_accuracy'][-1]
                    bar.increment()
        bar.finish()

        return params[accuracies.index(max(accuracies))]

    def train(self, encoded_data):
        X, y = encoded_data['X'], encoded_data['y']

        best_params = self._tune_hyperparameters(X, y)
        print('Best hyperparameters:')
        for key, value in best_params.items():
            print(f'  - {key}: {value}')

        print('Training best hypermodel...')
        models = []
        accuracies = []
        for _ in range(5):
            model = self._build_model(best_params)
            model.fit(
                X,
                y,
                verbose=0,
                epochs=best_params['epochs']
            )
            metrics_train = model.evaluate(
                X,
                y,
                verbose=0,
                return_dict=True
            )

            models.append(model)
            accuracies.append(metrics_train['accuracy'])

        self.model = models[accuracies.index(max(accuracies))]
        self.evaluate(encoded_data)

    def evaluate(self, encoded_data):
        X, y = encoded_data['X'], encoded_data['y']

        metrics = self.model.evaluate(
            X,
            y,
            verbose=0,
            return_dict=True
        )

        print('Model metrics:')
        for key, value in metrics.items():
            print(f'  - {key}: {value:.4f}')

    def predict(self, encoded_data):
        names, X = encoded_data['names'], encoded_data['X']
        y = self.model.predict(X, verbose=0)[:, 1]

        predictions = ['# RNA\tProtein\tLabel\tScore\n']
        for name, pred in zip(names, y):
            predictions.append(
                f'{name[0]}\t'
                f'{name[1]}\t'
                f'{"Interaction" if pred > 0.5 else "NoInteraction"}\t'
                f'{pred:.4f}\n'
            )

        return predictions

    def save(self, output_file_name):
        self.model.save(output_file_name, save_format='h5')

    def load(self, input_file_name):
        self.model = tf.keras.models.load_model(input_file_name)

class MIRLOEncoder(LPIDataEncoder):
    def encode(self, data):
        if 'predict.LION' in data:
            names, X = self._get_names_and_features(data['predict.LION'])
            return {'names': names, 'X': X}
        elif 'train.LION' in data and 'train.labels' in data:
            _, X = self._get_names_and_features(data['train.LION'])
            y = tf.keras.utils.to_categorical(
                data['train.labels'],
                num_classes=2
            )
            return {'X': X, 'y': y}
        elif 'test.LION' in data and 'test.labels' in data:
            _, X = self._get_names_and_features(data['test.LION'])
            y = tf.keras.utils.to_categorical(
                data['test.labels'],
                num_classes=2
            )
            return {'X': X, 'y': y}
        raise ValueError('EROR: Invalid data provided to the encoder')

    def _get_names_and_features(self, data):
        names, X = [], []
        for line in data:
            line = line.split()
            names.append([line[0], line[1]])
            X.append(np.array([float(f) for f in line[2:]]))
        return names, np.array(X)

class LIONProvider(LPIDataProvider):
    def __init__(self, rnas, proteins):
        self.rnas = rnas
        self.proteins = proteins

    def load(self):
        print('Extracting features from LION...')
        print(f'  - RNAs: {self.rnas}')
        print(f'  - proteins: {self.proteins}')
        script_path = os.path.dirname(os.path.realpath(__file__))
        Rscript_path = os.path.join(
            script_path,
            'scripts',
            'LION_feature_extract.R'
        )
        tmp_file_name = next(tempfile._get_candidate_names())

        process = subprocess.Popen(
            f'Rscript {Rscript_path}'
            f' {self.rnas}'
            f' {self.proteins}'
            f' {tmp_file_name}'.split(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if process.wait() != 0:
            raise ValueError('ERROR: LION feature extraction failed')

        lines = []
        with open(tmp_file_name, 'r') as f:
            lines = f.readlines()
        os.remove(tmp_file_name)

        return lines

if __name__ == '__main__':
    parser = ArgumentParser(
        prog='MIRLO',
        description='MIRLO: siMple, straIght foRward, and accurate LncRNA-prOtein prediction'
    )

    subparsers = parser.add_subparsers(dest='mode', required=True, help='run modes')

    train_parser = subparsers.add_parser('train', help='train a model')
    train_parser.add_argument('rnas', help='lncRNA sequences in FASTA format')
    train_parser.add_argument('proteins', help='protein sequences in FASTA format')
    train_parser.add_argument('labels', help='labels for the lncRNA-protein pairs')
    train_parser.add_argument('-o', '--output', required=False, default='MIRLO.h5', help='output file name for the trained model')

    evaluate_parser = subparsers.add_parser('evaluate', help='evaluate a model')
    evaluate_parser.add_argument('rnas', help='lncRNA sequences in FASTA format')
    evaluate_parser.add_argument('proteins', help='protein sequences in FASTA format')
    evaluate_parser.add_argument('labels', help='labels for the lncRNA-protein pairs')
    evaluate_parser.add_argument('model', help='trained model')

    predict_parser = subparsers.add_parser('predict', help='predict lncRNA-protein interactions using a model')
    predict_parser.add_argument('rnas', help='lncRNA sequences in FASTA format')
    predict_parser.add_argument('proteins', help='protein sequences in FASTA format')
    predict_parser.add_argument('model', help='trained model')
    predict_parser.add_argument('-o', '--output', required=False, default='MIRLO_predictions.txt', help='output file name for the predictions')

    args = parser.parse_args()

    if args.mode == 'train':
        data_providers = {
            'train.LION': LIONProvider(args.rnas, args.proteins),
            'train.labels': LPILabelsProvider(args.labels)
        }
    elif args.mode == 'evaluate':
        data_providers = {
            'test.LION': LIONProvider(args.rnas, args.proteins),
            'test.labels': LPILabelsProvider(args.labels)
        }
    elif args.mode == 'predict':
        data_providers = {
            'predict.LION': LIONProvider(args.rnas, args.proteins)
        }
    else:
        raise ValueError('ERROR: Invalid run mode')

    predictor = LPIPredictor(
        name='MIRLO',
        model=MIRLOModel(),
        encoder=MIRLOEncoder(),
        data_providers=data_providers,
        model_file=args.model if args.mode != 'train' else None,
        output_file=args.output if args.mode != 'evaluate' else None
    )
    predictor.run(args.mode)
