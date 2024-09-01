% training set
function [input_train, output_train, input_test, output_test] = training_set(input, output, train_ratio_time, train_ratio_neuron)
num_observations = size(input, 1);
num_neruons = size(input, 2);

if train_ratio_time == 1
    train_indices_time=1:num_observations;
else
    % random time
    random_indices = randperm(num_observations);
    if train_ratio_time >1
        num_train = train_ratio_time;
    else
        num_train = round(train_ratio_time * num_observations);
    end
    train_indices_time = random_indices(1:num_train);
end

if train_ratio_neuron == 1
    train_indices_neuron = 1:num_neruons;
else
    % random neuron
    random_indices = randperm(num_neruons);
    if train_ratio_neuron >1
        num_train = train_ratio_neuron;
    else
        num_train = round(train_ratio_neuron * num_neruons);
    end
    train_indices_neuron = random_indices(1:num_train);
end

input_train = input(train_indices_time, train_indices_neuron);
output_train = output(train_indices_time);
input_test = input(:,train_indices_neuron);
output_test = output;
end