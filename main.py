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

from joblib import dump, load

from itertools import product

from argparse import ArgumentParser

from lpi_prediction import LPIModel
from lpi_prediction import LPIPredictor
from lpi_prediction import LPIDataEncoder
from lpi_prediction import LPIDataProvider
from lpi_prediction import LPILabelsProvider

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DO_CACHING = True

class MIRLOModel(LPIModel):
    def __init__(self):
        self.max_epochs = 500

    def _build_model(self, params):
        model = tf.keras.models.Sequential()

        for _ in range(params['layers']):
            model.add(tf.keras.layers.Dense(units=params['neurons']))
            model.add(tf.keras.layers.PReLU())
            model.add(tf.keras.layers.Dropout(rate=params['dropout']))
        model.add(tf.keras.layers.Dense(units=2, activation='softmax'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=params['learning_rate']
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        hyperparams = {
            'layers': [1],
            'neurons': [32, 64, 128],
            'patience': [10, 20, 30],
            'dropout': [0.0, 0.25, 0.5, 0.7],
            'learning_rate': [0.1, 0.01]
        }

        params = []
        for param_values in product(*hyperparams.values()):
            params.append(dict(zip(hyperparams.keys(), param_values)))
        accuracies = [0] * len(params)

        bar = progressbar.ProgressBar(
            min_value=0,
            max_value=len(params) * 5,
            widgets=[
                'Hyperparameter tuning... ',
                '(', progressbar.AbsoluteETA(), ')'
            ]
        ).start()

        if len(params) > 1:
            for i, param in enumerate(params):
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    restore_best_weights=True,
                    patience=param['patience']
                )
                for _ in range(10):
                    model = self._build_model(param)
                    model.fit(
                        X_train,
                        y_train,
                        verbose=0,
                        epochs=self.max_epochs,
                        callbacks=[early_stopping],
                        validation_data=(X_val, y_val)
                    )
                    metrics = model.evaluate(
                        X_val,
                        y_val,
                        verbose=0,
                        return_dict=True
                    )
                    accuracies[i] += metrics['accuracy']
                    bar.increment()
        bar.finish()

        return params[accuracies.index(max(accuracies))]

    def train(self, encoded_data):
        X, y = encoded_data['X'], encoded_data['y']

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            stratify=y,
            shuffle=True,
            test_size=.2
        )

        self.scaler = StandardScaler().fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)

        best_params = self._tune_hyperparameters(X_train, y_train, X_val, y_val)
        print('Best hyperparameters:')
        for key, value in best_params.items():
            print(f'  - {key}: {value}')

        print('Training best hypermodel...')

        models = []
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            restore_best_weights=True,
            patience=best_params['patience']
        )
        for _ in range(10):
            model = self._build_model(best_params)
            model.fit(
                X_train,
                y_train,
                verbose=0,
                epochs=self.max_epochs,
                callbacks=[early_stopping],
                validation_data=(X_val, y_val)
            )
            metrics_train = model.evaluate(
                X_train,
                y_train,
                verbose=0,
                return_dict=True
            )
            metrics_val = model.evaluate(
                X_val,
                y_val,
                verbose=0,
                return_dict=True
            )
            models.append(
                (
                    metrics_val['accuracy'],
                    metrics_val['loss'],
                    metrics_train['accuracy'],
                    metrics_train['loss'],
                    model
                )
            )

        self.model = sorted(models, key=lambda x: x[0])[-1][4]
        print('Model metrics:')
        print(f'  - loss: {models[-1][3]:.4f}')
        print(f'  - accuracy: {models[-1][2]:.4f}')
        print(f'  - val_loss: {models[-1][1]:.4f}')
        print(f'  - val_accuracy: {models[-1][0]:.4f}')

    def evaluate(self, encoded_data):
        X, y = encoded_data['X'], encoded_data['y']
        X = self.scaler.transform(X)

        metrics = self.model.evaluate(X, y, verbose=0, return_dict=True)

        print('Model metrics:')
        for key, value in metrics.items():
            print(f'  - {key}: {value:.4f}')

    def predict(self, encoded_data):
        names, X = encoded_data['names'], encoded_data['X']
        X = self.scaler.transform(X)
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
        self.model.save(f'{output_file_name}.h5', save_format='h5')
        dump(self.scaler, f'{output_file_name}.joblib')

    def load(self, input_file_name):
        self.model = tf.keras.models.load_model(f'{input_file_name}.h5')
        self.scaler = load(f'{input_file_name}.joblib')

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

        if DO_CACHING:
            import hashlib
            cache_name = hashlib.sha256(
                f'{self.rnas}_{self.proteins}'.encode('utf-8')
            ).hexdigest()

            if os.path.exists(f'cache/{cache_name}'):
                with open(f'cache/{cache_name}', 'r') as f:
                    lines = f.readlines()
                return lines

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

        if DO_CACHING:
            os.makedirs('cache', exist_ok=True)

            import hashlib
            cache_name = hashlib.sha256(
                f'{self.rnas}_{self.proteins}'.encode('utf-8')
            ).hexdigest()

            with open(f'cache/{cache_name}', 'w') as f:
                f.writelines(lines)

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
    train_parser.add_argument('-o', '--output', required=False, default='MIRLO', help='output file name for the trained model')

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
