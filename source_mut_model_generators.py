import os

import utils
import source_mut_operators
import network
import tensorflow as tf


class SourceMutatedModelGenerators():

    def __init__(self, model_architecture='FC'):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.utils = utils.GeneralUtils()
        self.model_architecture = model_architecture
        if self.model_architecture == 'CNN':
            self.network = network.CNNNetwork()
        else:
            self.network = network.FCNetwork()
        
        self.source_mut_opts = source_mut_operators.SourceMutationOperators()
        self.valid_modes = ['DR', 'LE', 'DM', 'DF', 'NP', 'LR', 'LAs', 'AFRs']
    

    def integration_test(self, verbose=False):
        modes = ['DR', 'LE', 'DM', 'DF', 'NP', 'LR', 'LAs', 'AFRs']

        # Model creation
        # This should variates according to the value of self.model_architecture
        train_dataset, test_dataset = self.network.load_data()
        if self.model_architecture == 'CNN':
            model = self.network.create_CNN_model_1()
        else: 
            model = self.network.create_normal_FC_model()

        # Test for generate_model_by_source_mutation function 
        for mode in modes:
            name_of_saved_file = mode + '_model'
            self.generate_model_by_source_mutation(train_dataset, test_dataset, model, mode, verbose=verbose)


    def generate_model_by_source_mutation(self, train_dataset, test_dataset, model, mode, mutation_ratio, verbose=False, save_model=True):
        mutated_datas, mutated_labels = None, None
        mutated_model = None
        assert mode in self.valid_modes, 'Input mode ' + mode + ' is not implemented'
        
        # Parameters can experiment with 
        suffix = '_model'
        name_of_saved_file = mode + suffix
        with_checkpoint = False
        mutated_layer_indices = None


        lower_bound = 0
        upper_bound = 9
        STD = 100

        if mode == 'DR':
            (mutated_datas, mutated_labels), mutated_model = self.source_mut_opts.DR_mut(train_dataset, model, mutation_ratio)
        elif mode == 'LE':
            (mutated_datas, mutated_labels), mutated_model = self.source_mut_opts.LE_mut(train_dataset, model, lower_bound, upper_bound, mutation_ratio)
        elif mode == 'DM':
            (mutated_datas, mutated_labels), mutated_model = self.source_mut_opts.DM_mut(train_dataset, model, mutation_ratio)
        elif mode == 'DF':
            (mutated_datas, mutated_labels), mutated_model = self.source_mut_opts.DF_mut(train_dataset, model, mutation_ratio)
        elif mode == 'NP':
            (mutated_datas, mutated_labels), mutated_model = self.source_mut_opts.NP_mut(train_dataset, model, mutation_ratio, STD=STD)
        elif mode == 'LR':
            (mutated_datas, mutated_labels), mutated_model = self.source_mut_opts.LR_mut(train_dataset, model, mutated_layer_indices=mutated_layer_indices)
        elif mode == 'LAs':
            (mutated_datas, mutated_labels), mutated_model = self.source_mut_opts.LAs_mut(train_dataset, model, mutated_layer_indices=mutated_layer_indices)
        elif mode == 'AFRs':
            (mutated_datas, mutated_labels), mutated_model = self.source_mut_opts.AFRs_mut(train_dataset, model, mutated_layer_indices=mutated_layer_indices)
        else:
            pass 

        mutated_model = self.network.compile_model(mutated_model)
        trained_mutated_model = self.network.train_model(mutated_model, mutated_datas, mutated_labels, with_checkpoint=with_checkpoint)

        test_datas, test_labels = test_dataset

        acc_trained_mutated_model = trained_mutated_model.evaluate(test_datas, test_labels)[1]  

        if save_model:
            self.network.save_model(trained_mutated_model, name_of_saved_file, mode)
        return acc_trained_mutated_model

    def generate_model_by_source_mutation_metrics(self, train_dataset, test_dataset, model, mode, mutation_ratio, verbose=False, save_model=True):
        mutated_datas, mutated_labels = None, None
        mutated_model = None
        assert mode in self.valid_modes, 'Input mode ' + mode + ' is not implemented'
        
        # Parameters can experiment with 
        suffix = '_model'
        name_of_saved_file = mode + suffix
        with_checkpoint = False
        mutated_layer_indices = None


        lower_bound = 0
        upper_bound = 9
        STD = 100

        if mode == 'DR':
            (mutated_datas, mutated_labels), mutated_model = self.source_mut_opts.DR_mut(train_dataset, model, mutation_ratio)
        elif mode == 'LE':
            (mutated_datas, mutated_labels), mutated_model = self.source_mut_opts.LE_mut(train_dataset, model, lower_bound, upper_bound, mutation_ratio)
        elif mode == 'DM':
            (mutated_datas, mutated_labels), mutated_model = self.source_mut_opts.DM_mut(train_dataset, model, mutation_ratio)
        elif mode == 'DF':
            (mutated_datas, mutated_labels), mutated_model = self.source_mut_opts.DF_mut(train_dataset, model, mutation_ratio)
        elif mode == 'NP':
            (mutated_datas, mutated_labels), mutated_model = self.source_mut_opts.NP_mut(train_dataset, model, mutation_ratio, STD=STD)
        elif mode == 'LR':
            (mutated_datas, mutated_labels), mutated_model = self.source_mut_opts.LR_mut(train_dataset, model, mutated_layer_indices=mutated_layer_indices)
        elif mode == 'LAs':
            (mutated_datas, mutated_labels), mutated_model = self.source_mut_opts.LAs_mut(train_dataset, model, mutated_layer_indices=mutated_layer_indices)
        elif mode == 'AFRs':
            (mutated_datas, mutated_labels), mutated_model = self.source_mut_opts.AFRs_mut(train_dataset, model, mutated_layer_indices=mutated_layer_indices)
        else:
            pass 

        mutated_model = self.network.compile_model(mutated_model)
        mutated_model = self.network.train_model(mutated_model, mutated_datas, mutated_labels, with_checkpoint=with_checkpoint)

        test_datas, test_labels = test_dataset

        # acc_trained_mutated_model = trained_mutated_model.evaluate(test_datas, test_labels)[1]  

        # if save_model:
        #     self.network.save_model(trained_mutated_model, name_of_saved_file, mode)
        # return acc_trained_mutated_model

        test_results = mutated_model.predict(test_datas)
        correct_indices = [i for i in range(len(test_results)) if test_results[i].argmax() == test_labels[i].argmax()]
        correct_labels = test_labels[correct_indices].argmax(axis=1)
        incorrect_indices = [i for i in range(len(test_results)) if test_results[i].argmax() != test_labels[i].argmax()]
        incorrect_labels = test_labels[incorrect_indices].argmax(axis=1)
        killed_classes = list(set(incorrect_labels))
        killed_classes = [int(item) for item in killed_classes]
        accuracy = len(correct_indices) / len(test_results)
        accuracy_per_class = [len([j for j in correct_labels if j == i])/len([j for j in test_labels.argmax(axis=1) if j == i]) for i in range(len(test_labels[0]))]
        return {'accuracy': accuracy, 'accuracy_per_class': accuracy_per_class, 'killed_classes': killed_classes}