import source_mut_model_generators
import model_mut_model_generators
import network
import json
import numpy as np
import utils

class RunMutants():
    def __init__(self, model_name='FC', repetition_num=10, mutation_ratios=[0.1], test_from="test", test_num=1000, test_uniform=True, from_checkpoint=False):
        self.utils = utils.GeneralUtils()
        self.model_name = model_name
        if self.model_name not in ['FC', 'CNN1', 'CNN2', 'VGG', 'ResNet']:
            raise ValueError('model_name should be either FC, CNN1, CNN2, VGG, or ResNet')
        if self.model_name == 'CNN1':
            self.source_mut_model_generators = source_mut_model_generators.SourceMutatedModelGenerators(model_architecture='CNN')
            self.model_mut_model_generators = model_mut_model_generators.ModelMutatedModelGenerators(model_architecture='CNN')
            self.network = network.CNNNetwork()
            self.model = self.network.create_CNN_model_1()
        elif self.model_name == 'CNN2':
            self.source_mut_model_generators = source_mut_model_generators.SourceMutatedModelGenerators(model_architecture='CNN')
            self.model_mut_model_generators = model_mut_model_generators.ModelMutatedModelGenerators(model_architecture='CNN')
            self.network = network.CNNNetwork()
            self.model = self.network.create_CNN_model_2()
        elif self.model_name == "VGG":
            self.source_mut_model_generators = source_mut_model_generators.SourceMutatedModelGenerators(model_architecture='CNN')
            self.model_mut_model_generators = model_mut_model_generators.ModelMutatedModelGenerators(model_architecture='CNN')
            self.network = network.CNNNetwork()
            self.model = self.network.create_VGG16_model()
        elif self.model_name == "ResNet":
            self.source_mut_model_generators = source_mut_model_generators.SourceMutatedModelGenerators(model_architecture='CNN')
            self.model_mut_model_generators = model_mut_model_generators.ModelMutatedModelGenerators(model_architecture='CNN')
            self.network = network.CNNNetwork()
            self.model = self.network.create_resnet50_model()
        else:
            self.source_mut_model_generators = source_mut_model_generators.SourceMutatedModelGenerators(model_architecture='FC')
            self.model_mut_model_generators = model_mut_model_generators.ModelMutatedModelGenerators(model_architecture='FC')
            self.network = network.FCNetwork()
            self.model = self.network.create_normal_FC_model()
        self.train_dataset, self.test_dataset = self.network.load_data()
        (self.train_datas, self.train_labels), (self.test_datas, self.test_labels) = self.train_dataset, self.test_dataset
        self.compiled_model = self.network.compile_model(self.model)
        if from_checkpoint:
            try:
                self.trained_model = self.network.load_model(self.model_name)
                print('Loaded trained model from checkpoint')
            except Exception as e:
                print('Failed to load model from checkpoint, training a new model. Error:', e)
                self.trained_model = self.network.train_model(self.compiled_model, self.train_datas, self.train_labels)
                self.network.save_model(self.trained_model, self.model_name)
        else:
            self.trained_model = self.network.train_model(self.compiled_model, self.train_datas, self.train_labels)
            self.network.save_model(self.trained_model, self.model_name)
        self.acc_trained_model = self.trained_model.evaluate(self.test_datas, self.test_labels)[1]
        self.repetition_num = repetition_num
        self.mutation_ratios = mutation_ratios
        self.records_filename = self.model_name + "_records_colab.json"
        if from_checkpoint:
            try:
                with open(self.records_filename, 'r') as f:
                    self.records = json.load(f)
                print('Loaded records from checkpoint')
            except Exception as e:
                print('Error when loading records from checkpoint:', e, '. Starting from scratch.')
                self.records = dict()
                self.records[str(('raw'))] = [self.acc_trained_model]
        else:
            self.records = dict()
            self.records[str(('raw'))] = [self.acc_trained_model]
        self.test_from = test_from
        self.test_num = test_num
        self.test_uniform = test_uniform
        self.test_prime_datas, self.test_prime_labels = self.get_test_prime_dataset(self.test_datas, self.test_labels)
        self.train_prime_datas, self.train_prime_labels = self.get_train_prime_dataset(self.train_datas, self.train_labels)

    def get_test_prime_dataset(self, test_datas, test_labels):
        test_results = self.trained_model.predict(test_datas)
        correct_indices = [i for i in range(len(test_results)) if test_results[i].argmax() == test_labels[i].argmax()]
        # print('Number of correct predictions:', len(correct_indices), '/', len(test_results))
        # test_prime_label = test_labels
        # return test_prime_data, test_prime_label
        return test_datas[correct_indices], test_labels[correct_indices]
    
    def get_train_prime_dataset(self, train_datas, train_labels):
        train_results = self.trained_model.predict(train_datas)
        correct_indices = [i for i in range(len(train_results)) if train_results[i].argmax() == train_labels[i].argmax()]
        return train_datas[correct_indices], train_labels[correct_indices]

    def run_vanilla_model(self):
        print('------------- Start running vanilla model -------------')
        for i in range(1, self.repetition_num + 1):
            key = str(('raw'))
            if key not in self.records:
                self.records[key] = []
            else:
                if len(self.records[key]) >= i:
                    print('Already run this experiment:', key, i)
                    continue
            curr_network = network.FCNetwork() if self.model_name == 'FC' else network.CNNNetwork()
            model = curr_network.create_normal_FC_model() if self.model_name == 'FC' else curr_network.create_CNN_model_1() if self.model_name == 'CNN1' else curr_network.create_CNN_model_2()
            compiled_model = curr_network.compile_model(model)
            trained_model = curr_network.train_model(compiled_model, self.train_datas, self.train_labels)
            acc_vanilla_model = trained_model.evaluate(self.test_datas, self.test_labels)[1]
            print(self.model_name, '- Vanilla model - Repetition:', i, '/', self.repetition_num, 'Accuracy', acc_vanilla_model)
            self.records[key].append(acc_vanilla_model)
            with open(self.records_filename, 'w') as f:
                json.dump(self.records, f)
        print('------------- Finished running vanilla model -------------')
    
    def run_source_mutants(self):
        print('------------- Start running source mutants -------------')
        for mutation_ratio in self.mutation_ratios:
            for k, mode in enumerate(self.source_mut_model_generators.valid_modes):
                for i in range(1, self.repetition_num + 1):
                    key = str(('source', mutation_ratio, mode, self.test_from, self.test_num, self.test_uniform))
                    if key not in self.records:
                        self.records[key] = []
                    else:
                        if len(self.records[key]) >= i:
                            print('Already run this experiment:', key, i)
                            continue
                    if self.test_from == 'test':
                        if not self.test_uniform:
                            random_indices = self.utils.get_random_indices_non_uniform(self.test_prime_labels, self.test_num)
                        else:
                            random_indices = np.random.choice(len(self.test_prime_datas), self.test_num, replace=False)
                        test_datas = self.test_prime_datas[random_indices]
                        test_labels = self.test_prime_labels[random_indices]
                        results = self.source_mut_model_generators.generate_model_by_source_mutation_metrics(train_dataset=self.train_dataset, test_dataset=(test_datas, test_labels), model=self.model, mode=mode, mutation_ratio=mutation_ratio, verbose=False, save_model=False)
                    else:
                        if not self.test_uniform:
                            random_indices = self.utils.get_random_indices_non_uniform(self.train_prime_labels, self.test_num)
                        else:
                            random_indices = np.random.choice(len(self.train_prime_datas), self.test_num, replace=False)
                        train_datas = self.train_prime_datas[random_indices]
                        train_labels = self.train_prime_labels[random_indices]
                        results = self.source_mut_model_generators.generate_model_by_source_mutation_metrics(train_dataset=self.train_dataset, test_dataset=(train_datas, train_labels), model=self.model, mode=mode, mutation_ratio=mutation_ratio, verbose=False, save_model=False)
                    # print(results)
                    print(self.model_name, '- Source mutants - Mutation Ratio:', mutation_ratio, 'Mode: ', mode, '(', k + 1, '/', len(self.model_mut_model_generators.valid_modes), ')', 'Repetition:', i, '/', self.repetition_num, 'Accuracy', results['accuracy'], 'test_from', self.test_from, 'test_num:', self.test_num, 'test_uniform:', self.test_uniform)
                    self.records[key].append(results)
                    with open(self.records_filename, 'w') as f:
                        json.dump(self.records, f)
        print('------------- Finished running source mutants -------------')

    def run_model_mutants(self):
        print('------------- Start running model mutants -------------')
        for mutation_ratio in self.mutation_ratios:
            for k, mode in enumerate(self.model_mut_model_generators.valid_modes):
                for i in range(1, self.repetition_num + 1):
                    key = str(('model', mutation_ratio, mode, self.test_from, self.test_num, self.test_uniform))
                    if key not in self.records:
                        self.records[key] = []
                    else:
                        if len(self.records[key]) >= i:
                            print('Already run this experiment:', key, i)
                            continue
                    if self.test_from == 'test':
                        if not self.test_uniform:
                            random_indices = self.utils.get_random_indices_non_uniform(self.test_prime_labels, self.test_num)
                        else:
                            random_indices = np.random.choice(len(self.test_prime_datas), self.test_num, replace=False)
                        test_datas = self.test_prime_datas[random_indices]
                        test_labels = self.test_prime_labels[random_indices]
                        results = self.model_mut_model_generators.generate_model_by_model_mutation_metrics(model=self.trained_model, mode=mode, mutation_ratio=mutation_ratio, test_datas=test_datas, test_labels=test_labels)
                    else:
                        if not self.test_uniform:
                            random_indices = self.utils.get_random_indices_non_uniform(self.train_prime_labels, self.test_num)
                        else:
                            random_indices = np.random.choice(len(self.train_prime_datas), self.test_num, replace=False)
                        train_datas = self.train_prime_datas[random_indices]
                        train_labels = self.train_prime_labels[random_indices]
                        results = self.model_mut_model_generators.generate_model_by_model_mutation_metrics(model=self.trained_model, mode=mode, mutation_ratio=mutation_ratio, test_datas=train_datas, test_labels=train_labels)
                    # print(results)
                    # return
                    print(self.model_name, '- Model mutants - Mutation Ratio:', mutation_ratio, 'Mode: ', mode, '(', k + 1, '/', len(self.model_mut_model_generators.valid_modes), ')', 'Repetition:', i, '/', self.repetition_num, 'Accuracy', results['accuracy'], 'test_from:', self.test_from, 'test_num:', self.test_num, 'test_uniform:', self.test_uniform)
                    self.records[key].append(results)
                    with open(self.records_filename, 'w') as f:
                        json.dump(self.records, f)
        print('------------- Finished running model mutants -------------')

if __name__ == '__main__':
    # mutation_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # repetition_num = 10dawsdas
    mutation_ratios = [0.01]
    repetition_num = 10
    # test_from = 'test'
    # test_num = 1000

    while True:
        # try:
            for test_from, test_num in zip(['train', 'test'], [5000, 1000]):
                # for test_from, test_num in zip(['test', 'train'], [1000, 5000]):
                for test_uniform in [True, False]:
                    # FC
                    # print('FC', test_from, test_num, test_uniform)
                    # run_mutants = RunMutants(model_name='FC', repetition_num=repetition_num, mutation_ratios=mutation_ratios, test_from=test_from, test_num=test_num, test_uniform=test_uniform, from_checkpoint=True)
                    # # run_mutants.run_vanilla_model()
                    # # run_mutants.run_model_mutants()
                    # run_mutants.run_source_mutants()

                    # CNN1
                    # print('CNN1', test_from, test_num, test_uniform)
                    # run_mutants = RunMutants(model_name='CNN1', repetition_num=repetition_num, mutation_ratios=mutation_ratios, test_from=test_from, test_num=test_num, test_uniform=test_uniform, from_checkpoint=True)
                    # # run_mutants.run_vanilla_model()
                    # # run_mutants.run_model_mutants()
                    # run_mutants.run_source_mutants()

                    # CNN2
                    # print('CNN2', test_from, test_num, test_uniform)
                    # run_mutants = RunMutants(model_name='CNN2', repetition_num=repetition_num, mutation_ratios=mutation_ratios, test_from=test_from, test_num=test_num, test_uniform=test_uniform, from_checkpoint=True)
                    # # run_mutants.run_vanilla_model()
                    # # run_mutants.run_model_mutants()
                    # run_mutants.run_source_mutants()

                    # VGG
                    # print('VGG', test_from, test_num, test_uniform)
                    # run_mutants = RunMutants(model_name='VGG', repetition_num=repetition_num, mutation_ratios=mutation_ratios, test_from=test_from, test_num=test_num, test_uniform=test_uniform, from_checkpoint=True)
                    # run_mutants.run_vanilla_model()
                    # # run_mutants.run_model_mutants()
                    # run_mutants.run_source_mutants()

                    # ResNet
                    print('ResNet', test_from, test_num, test_uniform)
                    run_mutants = RunMutants(model_name='ResNet', repetition_num=repetition_num, mutation_ratios=mutation_ratios, test_from=test_from, test_num=test_num, test_uniform=test_uniform, from_checkpoint=True)
                    run_mutants.run_vanilla_model()
                    run_mutants.run_model_mutants()
                    run_mutants.run_source_mutants()
            break

        # except Exception as e:
        #     print('Error:', e)
        #     print('Restarting...')
        #     # break
    
    # print('Run Mutants Finished!')
    # run_mutants = RunMutants(model_name='FC', repetition_num=10, mutation_ratios=mutation_ratios, from_checkpoint=True)
    # run_mutants.get_test_prime_data(run_mutants.test_datas, run_mutants.test_labels)