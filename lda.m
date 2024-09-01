% LDA
function [explained_variance, V_sorted]=lda(X, labels)
num_neuron = size(X,2);
% class means
classes = unique(labels);
class_means =zeros(length(classes),num_neuron);
for i = 1:length(classes)
    class_means(i,:) = mean(X(labels==classes(i), :), 1);
end
% within-class scatter matrix Sw
Sw = zeros(num_neuron, num_neuron);
for i = 1:length(classes)
    Xi = X(labels==classes(i), :);
    Xi_centered = Xi - class_means(i, :);
    Sw = Sw + (Xi_centered' * Xi_centered);
end
% between-class scatter matrix Sb
overall_mean = mean(X, 1);
Sb = zeros(num_neuron, num_neuron);
for i = 1:length(classes)
    ni = size(X(labels==classes(i), :), 1);
    mean_diff = class_means(i, :) - overall_mean;
    Sb = Sb + ni * (mean_diff' * mean_diff);
end
% significant components
[V, D] = eig(Sb,Sw);
[eigenvalues_sorted, idx] = sort(diag(D), 'descend');
explained_variance = (eigenvalues_sorted / sum(eigenvalues_sorted)) * 100;
V_sorted = V(:, idx);
end