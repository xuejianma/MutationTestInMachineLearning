import os

import source_mut_operators
import utils, network, model_mut_operators

import keras 


class ModelMutatedModelGenerators():

    def __init__(self, model_architecture='FC'):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.utils = utils.GeneralUtils()
        self.model_utils = utils.ModelUtils()
        self.model_architecture = model_architecture
        if self.model_architecture == 'CNN':
            self.network = network.CNNNetwork()
        else:
            self.network = network.FCNetwork()
        
        self.model_mut_opts = model_mut_operators.ModelMutationOperators()
        (_, _), (test_datas, test_results) = self.network.load_data()
        self.test_datas = test_datas
        self.test_results = test_results
        self.valid_modes = ['GF', 'WS', 'NEB', 'NAI', 'NS', 'LD', 'LAm', 'AFRm']
         
    def integration_test(self, verbose=False):
        # Parameters 
        mutation_ratio = 0.2 
        # mutated_layer_indices = [0, 7]
        mutated_layer_indices = None
        modes = ['GF', 'WS', 'NEB', 'NAI', 'NS', 'LD', 'LAm', 'AFRm']
        # modes = ['LD']

        if self.model_architecture == 'CNN':        
            model = self.network.load_model('CNN_model2')
        else:
            model = self.network.load_model('normal_FC_model')

        for mode in modes:
            name_of_saved_file = mode + '_model'
            self.generate_model_by_model_mutation(model, mode, mutation_ratio, name_of_saved_file=name_of_saved_file, mutated_layer_indices=mutated_layer_indices, verbose=verbose) 

    def generate_model_by_model_mutation(self, model, mode, mutation_ratio, name_of_saved_file='mutated_model', mutated_layer_indices=None, STD=0.1, verbose=False, save_model=True):
        mutated_model = None
        assert mode in self.valid_modes, 'Input mode ' + mode + ' is not implemented'
        
        if mode == 'GF':
            mutated_model = self.model_mut_opts.GF_mut(model, mutation_ratio, prob_distribution='normal', STD=STD, mutated_layer_indices=mutated_layer_indices)
        elif mode == 'WS':
            mutated_model = self.model_mut_opts.WS_mut(model, mutation_ratio, mutated_layer_indices=mutated_layer_indices) 
        elif mode == 'NEB':
            mutated_model = self.model_mut_opts.NEB_mut(model, mutation_ratio, mutated_layer_indices=mutated_layer_indices)
        elif mode == 'NAI':
            mutated_model = self.model_mut_opts.NAI_mut(model, mutation_ratio, mutated_layer_indices=mutated_layer_indices)
        elif mode == 'NS':
            mutated_model = self.model_mut_opts.NS_mut(model, mutation_ratio, mutated_layer_indices=mutated_layer_indices)
        elif mode == 'LD':
            mutated_model = self.model_mut_opts.LD_mut(model, mutated_layer_indices=mutated_layer_indices)
        elif mode == 'LAm':
            mutated_model = self.model_mut_opts.LAm_mut(model, mutated_layer_indices=mutated_layer_indices)
        elif mode == 'AFRm':
            mutated_model = self.model_mut_opts.AFRm_mut(model, mutated_layer_indices=mutated_layer_indices)
        else:
            pass

        mutated_model = self.network.compile_model(mutated_model)
        acc_trained_mutated_model = mutated_model.evaluate(self.test_datas, self.test_results)[1]

        if verbose:
            self.model_utils.print_comparision_of_layer_weights(model, mutated_model)
            self.utils.print_messages_MMM_generators(mode, network=self.network, test_datas=self.test_datas, test_labels=self.test_results, model=model, mutated_model=mutated_model, STD=STD, mutation_ratio=mutation_ratio) 

        if save_model:
            self.network.save_model(mutated_model, name_of_saved_file, mode) 
        return acc_trained_mutated_model

    def generate_model_by_model_mutation_metrics(self, model, mode, mutation_ratio, test_datas, test_labels, name_of_saved_file='mutated_model', mutated_layer_indices=None, STD=0.1):
        mutated_model = None
        assert mode in self.valid_modes, 'Input mode ' + mode + ' is not implemented'
        
        if mode == 'GF':
            mutated_model = self.model_mut_opts.GF_mut(model, mutation_ratio, prob_distribution='normal', STD=STD, mutated_layer_indices=mutated_layer_indices)
        elif mode == 'WS':
            mutated_model = self.model_mut_opts.WS_mut(model, mutation_ratio, mutated_layer_indices=mutated_layer_indices) 
        elif mode == 'NEB':
            mutated_model = self.model_mut_opts.NEB_mut(model, mutation_ratio, mutated_layer_indices=mutated_layer_indices)
        elif mode == 'NAI':
            mutated_model = self.model_mut_opts.NAI_mut(model, mutation_ratio, mutated_layer_indices=mutated_layer_indices)
        elif mode == 'NS':
            mutated_model = self.model_mut_opts.NS_mut(model, mutation_ratio, mutated_layer_indices=mutated_layer_indices)
        elif mode == 'LD':
            mutated_model = self.model_mut_opts.LD_mut(model, mutated_layer_indices=mutated_layer_indices)
        elif mode == 'LAm':
            mutated_model = self.model_mut_opts.LAm_mut(model, mutated_layer_indices=mutated_layer_indices)
        elif mode == 'AFRm':
            mutated_model = self.model_mut_opts.AFRm_mut(model, mutated_layer_indices=mutated_layer_indices)
        else:
            pass

        mutated_model = self.network.compile_model(mutated_model)
        test_results = mutated_model.predict(test_datas)
        correct_indices = [i for i in range(len(test_results)) if test_results[i].argmax() == test_labels[i].argmax()]
        correct_labels = test_labels[correct_indices].argmax(axis=1)
        incorrect_indices = [i for i in range(len(test_results)) if test_results[i].argmax() != test_labels[i].argmax()]
        incorrect_labels = test_labels[incorrect_indices].argmax(axis=1)
        killed_classes = list(set(incorrect_labels))
        killed_classes = [int(item) for item in killed_classes]
        accuracy = len(correct_indices) / len(test_results)
        # print(list(sorted(test_labels.argmax(axis=1))))
        accuracy_per_class = [len([j for j in correct_labels if j == i])/len([j for j in test_labels.argmax(axis=1) if j == i]) for i in range(len(test_labels[0]))]
        return {'accuracy': accuracy, 'accuracy_per_class': accuracy_per_class, 'killed_classes': killed_classes}