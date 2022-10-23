import source_mut_model_generators
import model_mut_model_generators
import network
import json

class RunMutants():
    def __init__(self, model_name='FC', repetition_num=10, mutation_ratios=[0.1], from_checkpoint=False):
        self.model_name = model_name
        if self.model_name not in ['FC', 'CNN1', 'CNN2']:
            raise ValueError('model_name should be either FC, CNN1 or CNN2')
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
        self.records_filename = self.model_name + "_records.json"
        if from_checkpoint:
            try:
                with open(self.records_filename, 'r') as f:
                    self.records = json.load(f)
                print('Loaded records from checkpoint')
            except Exception as e:
                print('Error when loading records from checkpoint:', e, '. Starting from scratch.')
                self.records = dict()
                self.records[str(('raw', 'raw', 'raw'))] = [self.acc_trained_model]
        else:
            self.records = dict()
            self.records[str(('raw', 'raw', 'raw'))] = [self.acc_trained_model]

    def run_vanilla_model(self):
        print('------------- Start running vanilla model -------------')
        for i in range(1, self.repetition_num + 1):
            key = str(('raw', 'raw', 'raw'))
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
                    key = str(('source', mutation_ratio, mode))
                    if key not in self.records:
                        self.records[key] = []
                    else:
                        if len(self.records[key]) >= i:
                            print('Already run this experiment:', key, i)
                            continue
                    acc_source_mut_model = self.source_mut_model_generators.generate_model_by_source_mutation(train_dataset=self.train_dataset, test_dataset=self.test_dataset, model=self.model, mode=mode, mutation_ratio=mutation_ratio, verbose=False, save_model=False)
                    print(self.model_name, '- Source mutants - Mutation Ratio:', mutation_ratio, 'Mode: ', mode, '(', k + 1, '/', len(self.model_mut_model_generators.valid_modes), ')', 'Repetition:', i, '/', self.repetition_num, 'Accuracy', acc_source_mut_model)
                    self.records[key].append(acc_source_mut_model)
                    with open(self.records_filename, 'w') as f:
                        json.dump(self.records, f)
        print('------------- Finished running source mutants -------------')

    def run_model_mutants(self):
        print('------------- Start running model mutants -------------')
        for mutation_ratio in self.mutation_ratios:
            for k, mode in enumerate(self.model_mut_model_generators.valid_modes):
                for i in range(1, self.repetition_num + 1):
                    key = str(('model', mutation_ratio, mode))
                    if key not in self.records:
                        self.records[key] = []
                    else:
                        if len(self.records[key]) >= i:
                            print('Already run this experiment:', key, i)
                            continue
                    acc_model_mut_model = self.model_mut_model_generators.generate_model_by_model_mutation(model=self.trained_model, mode=mode, mutation_ratio=mutation_ratio, verbose=False, save_model=False)
                    print(self.model_name, '- Model mutants - Mutation Ratio:', mutation_ratio, 'Mode: ', mode, '(', k + 1, '/', len(self.model_mut_model_generators.valid_modes), ')', 'Repetition:', i, '/', self.repetition_num, 'Accuracy', acc_model_mut_model)
                    self.records[key].append(acc_model_mut_model)
                    with open(self.records_filename, 'w') as f:
                        json.dump(self.records, f)
        print('------------- Finished running model mutants -------------')

if __name__ == '__main__':
    mutation_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    repetition_num = 10

    while True:
        try:
            # FC
            run_mutants = RunMutants(model_name='FC', repetition_num=repetition_num, mutation_ratios=mutation_ratios, from_checkpoint=True)
            run_mutants.run_vanilla_model()
            run_mutants.run_model_mutants()
            run_mutants.run_source_mutants()

            # CNN1
            run_mutants = RunMutants(model_name='CNN1', repetition_num=repetition_num, mutation_ratios=mutation_ratios, from_checkpoint=True)
            run_mutants.run_vanilla_model()
            run_mutants.run_model_mutants()
            run_mutants.run_source_mutants()

            # CNN2
            run_mutants = RunMutants(model_name='CNN2', repetition_num=repetition_num, mutation_ratios=mutation_ratios, from_checkpoint=True)
            run_mutants.run_vanilla_model()
            run_mutants.run_model_mutants()
            run_mutants.run_source_mutants()

            break
        except Exception as e:
            print('Error:', e)
            print('Restarting...')
    
    print('Run Mutants Finished!')
