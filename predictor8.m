% Final evaluation with processed data:

% WHOLE ANALYSIS
% projected space, regression curve, normal test for distance, 
% test for same distribution,
% TO DO: regrssion region

% TRAINING AND TESTING ANALYSIS
% accuracy
% TO DO: verified set


clear
%%
load hc_30_processed.mat
trial = 59; % 58, 59, 77
intensity_raw = hc_30(trial).traces;
speed = hc_30(trial).speed;
%%
load hc_31_processed.mat
trial = 1;
intensity_raw = hc_31(trial).spike;
speed = hc_31(trial).speed;
%% All steps

% eMA of intensity
intensity_all=zeros(size(intensity_raw));
for i =1:size(intensity_raw,2)
    intensity_all(:,i) = eMA(intensity_raw(:,i),40);
end

% Random input and testing set
[intensity_t, speed_t, input, output_d]=training_set(intensity_all, speed, 1, 450);
% % no random
% intensity_t = intensity_t(1:1000,:);
% speed_t = speed_t(1:1000,:);

speed_r = [min(speed_t), max(speed_t)];
speed_t = speed_t - min(speed_t);
speed_t = speed_t / max(speed_t);



% LDA
speed_q1=round(1 + speed_t * (10 - 1));
[ex, V]=lda(intensity_t,speed_q1);
intensity_p = intensity_t * V(:, 1:3);

% testing set and projection
% input = intensity_all;
% output_d = speed;
input_p = input * V(:, 1:3);

% sepration by speed
num_group=100;
speed_q2 = round(1 + speed_t * (num_group - 1));
speed_q3 = linspace(0,1,num_group);
intensity_g={};
for i=1:num_group
    intensity_g{1,i}=intensity_p(speed_q2==i,:);
end
valid_ind = find(~cellfun(@isempty, intensity_g));
intensity_g = intensity_g(valid_ind);
speed_q3=speed_q3(valid_ind);
mean_group_valid = length(valid_ind);

% mean
intensity_mean=zeros(mean_group_valid,3);
for n=1:mean_group_valid
    for i=1:3
        intensity_mean(n,i) = mean(remove_outliers(intensity_g{1,n}(:,i)));
        % intensity_mean(n,i) = mean(intensity_g{1,n}(:,i));
    end
end

% projected means regression
x_fit = fit(speed_q3', intensity_mean(:,1), 'poly4');
y_fit = fit(speed_q3', intensity_mean(:,2), 'poly4');
z_fit = fit(speed_q3', intensity_mean(:,3), 'poly4');
x = @(t) feval(x_fit, t);
y = @(t) feval(y_fit, t);
z = @(t) feval(z_fit, t);

% distances
distances = abs(intensity_p-[x(speed_t), y(speed_t), z(speed_t)]);
distances_g=cell(1,num_group);
for i=1:num_group
    distances_g{1,i}=distances(speed_q2==i,:);
end
distances_std=zeros(mean_group_valid,3);
distances_mean=zeros(mean_group_valid,3);
distance_std=zeros(mean_group_valid,1);
distance_mean=zeros(mean_group_valid,1);
distance_g=cell(1,num_group);
for n=1:mean_group_valid
    distance_g{1,n} = sqrt(distances_g{1,n}(:,1).^2+distances_g{1,n}(:,2).^2+distances_g{1,n}(:,3).^2);
    distance_std(n,1) = std(distance_g{1,n});
    distance_mean(n,1) = mean(distance_g{1,n});
    for i=1:3
        distances_std(n,i) = std(distances_g{1,n}(:,i));
        distances_mean(n,i) = mean(distances_g{1,n}(:,i));
    end
end




% predict
output_e = [];



% % with extended curve
% tic;
% output_raw=zeros(size(input,1),1);
% for i = 1:size(output_raw,1)
%     F = @(t) (x(t) - input_p(i,1)).^2 + (y(t) - input_p(i,2)).^2 + (z(t) - input_p(i,3)).^2;
%     output_raw(i,1) = fminbnd(F, -0.1, 1.1);
% end
% output_raw(output_raw < 0) = 0;
% output_raw(output_raw > 1) = 1;
% output_e = [output_e, output_raw * (speed_r(2) - speed_r(1)) + speed_r(1)];
% fprintf('Reference:\n')
% toc;

% with eMA filter and extended curve
tic;
input_f=zeros(size(input_p));
for i =1:size(input_f,2)
    input_f(:,i) = eMA(input_p(:,i),2);
end
output_raw=zeros(size(input,1),1);
for i = 1:size(output_raw,1)
    F = @(t) (x(t) - input_f(i,1)).^2 + (y(t) - input_f(i,2)).^2 + (z(t) - input_f(i,3)).^2;
    output_raw(i,1) = fminbnd(F, -0.1, 1.1);
end
output_raw(output_raw < 0) = 0;
output_raw(output_raw > 1) = 1;
output_e = [output_e, output_raw * (speed_r(2) - speed_r(1)) + speed_r(1)];
fprintf('eMA filter:\n')
toc;


% accuracy
Result = table('Size', [size(output_e,2), 3], 'VariableTypes', {'double', 'double', 'double'}, ...
                       'VariableNames', {'corr', 'R2', 'RMSE'});
for i = 1:size(output_e,2)
    Result{i, 'corr'} = corr(output_e(:,i),output_d);
    Result{i, 'R2'} = fitlm(output_e(:,i),output_d).Rsquared.Ordinary;
    Result{i, 'RMSE'} = sqrt(mean((output_e(:,i) - output_d) .^ 2));
end
disp(Result);

% comparision plot
figure;
hold on
time_axis=linspace(0,(length(output_d)-1)/20,length(output_d));
plot(time_axis,output_d)
plot(time_axis,output_e)
legend
%%
% training figure
figure;
colormap jet
scatter3(intensity_p(:,1),intensity_p(:,2),intensity_p(:,3),2,categorical(speed_q2/100), 'filled');
xlabel('Feature 1');
ylabel('Feature 2');
zlabel('Feature 3');
title('3D Scatter Plot with Classes');
colorbar
grid on;

figure
hold on
t=linspace(0,1,1000);
lines(:,1)=x(t);
lines(:,2)=y(t);
lines(:,3)=z(t);
plot3(lines(:,1),lines(:,2),lines(:,3),'k','LineWidth',3)
%% testing figure
figure;
output_q = round(1 + output_d * (num_group - 1));
colormap jet
scatter3(input_p(:,1),input_p(:,2),input_p(:,3),2,categorical(output_q), 'filled');
xlabel('Feature 1');
ylabel('Feature 2');
zlabel('Feature 3');
title('3D Scatter Plot with Classes');
colorbar
grid on;
hold on
t=linspace(-0.1,1.1,1000);
plot3(x(t),y(t),z(t),'k','LineWidth',3)


%% noraml distribution test
p1 = zeros(numel(intensity_g),3);
p2 = zeros(numel(intensity_g),3);
h1 = zeros(numel(intensity_g),3);
h2 = zeros(numel(intensity_g),3);
for n = 1:numel(intensity_g)-1
for i=1:3
    data = intensity_g{1,n}(:,i);
    % Q1 = quantile(data, 0.25);
    % Q3 = quantile(data, 0.75);
    % IQR = Q3 - Q1;
    % lowerBound = Q1 - 1.5 * IQR;
    % upperBound = Q3 + 1.5 * IQR;
    % inliers = (data >= lowerBound) & (data <= upperBound);
    % data = data(inliers);

    [h1(n,i), p1(n,i)] = lillietest(data);
    [h2(n,i), p2(n,i)] = adtest(data);
end
end
(100-sum(h1))/100
(100-sum(h2))/100
%% distibution test whether the distribution of point is independent to speed
n = numel(distances_g);  % Number of vectors
p_matrix = zeros(n, n);  % Use NaN for upper triangular entries

% Perform pair-based rank sum tests
h_per=zeros(1,3);
p_t = zeros(n,3);
h_t = zeros(n,3);
for t =1:3
    h_sum=0;
for i = 1:n
    vec1 = distances_g{i}(:,t);
    for j = i+1:n
        % Extract vectors
        vec2 = distances_g{j}(:,t);
        
        % Perform rank sum test
        [p,h] = ranksum(vec1, vec2);
        
        % Store p-value in the matrix
        p_matrix(i, j) = h;
        p_matrix(j, i) = h;  % For symmetry
        % p_matrix(i, j) = p;
        % p_matrix(j, i) = p;  % For symmetry
        h_sum=h_sum+h;
    end
    [p_t(i,t),h_t(i,t)] = ranksum(vec1, distances(:,t));
end

h_per(1,t)=h_sum/(n*(n-1)/2)*100;
fprintf('in Axis %d, %5.2f pairs come from different distribution\n',[t,h_per(1,t)])

figure;
h = heatmap(p_matrix, 'Colormap', parula, 'ColorbarVisible', 'on');
title('Rank Sum Test p-values');
xlabel('Group');
ylabel('Group');
h.XDisplayLabels = repmat({''}, 1, n);  % Remove x-axis numbers
h.YDisplayLabels = repmat({''}, n, 1);  % Remove y-axis numbers

% figure;
% Y = pdist(p_matrix);         % Compute pairwise distances between rows
% Z = linkage(Y, 'average');      % Perform hierarchical clustering
% order = optimalleaforder(Z, Y);
% reordered_corr_matrix = p_matrix(order, order);
% h=heatmap(reordered_corr_matrix, 'Colormap', parula, 'ColorbarVisible', 'on');
% title('Reordered Correlation Matrix');
% order_labels = arrayfun(@num2str, order, 'UniformOutput', false);
% h.XDisplayLabels = order_labels;
% h.YDisplayLabels = order_labels;
end
%% distibution test whether the distribution of point is independent to speed
n = numel(distance_g);  % Number of vectors
p_matrix = zeros(n, n);  % Use NaN for upper triangular entries

% Perform pair-based rank sum tests
h_per=0;
p_t = zeros(n,1);
h_t = zeros(n,1);
    h_sum=0;
for i = 1:n
    vec1 = distance_g{i};
    for j = i+1:n
        % Extract vectors
        vec2 = distance_g{j};
        
        % Perform rank sum test
        [p,h] = ranksum(vec1, vec2);
        
        % Store p-value in the matrix
        p_matrix(i, j) = h;
        p_matrix(j, i) = h;  % For symmetry
        % p_matrix(i, j) = p;
        % p_matrix(j, i) = p;  % For symmetry
        h_sum=h_sum+h;
    end
    [p_t(i,1),h_t(i,1)] = ranksum(vec1, distances(:,1));
end

h_per=h_sum/(n*(n-1)/2)*100;
fprintf('%5.2f pairs come from different distribution\n',h_per)

figure;
h = heatmap(p_matrix, 'Colormap', parula, 'ColorbarVisible', 'on');
title('Rank Sum Test p-values');
xlabel('Group');
ylabel('Group');
h.XDisplayLabels = repmat({''}, 1, n);  % Remove x-axis numbers
h.YDisplayLabels = repmat({''}, n, 1);  % Remove y-axis numbers

% figure;
% Y = pdist(p_matrix);         % Compute pairwise distances between rows
% Z = linkage(Y, 'average');      % Perform hierarchical clustering
% order = optimalleaforder(Z, Y);
% reordered_corr_matrix = p_matrix(order, order);
% h=heatmap(reordered_corr_matrix, 'Colormap', parula, 'ColorbarVisible', 'on');
% title('Reordered Correlation Matrix');
% order_labels = arrayfun(@num2str, order, 'UniformOutput', false);
% h.XDisplayLabels = order_labels;
% h.YDisplayLabels = order_labels;


%% 3D plot
% Define the parameter t
t = linspace(0, 1, 100);

% Parametric equations for the 3D curve
x_list = x(t);
y_list = y(t);
z_list = z(t);
dx = gradient(x_list, t);
dy = gradient(y_list, t);
dz = gradient(z_list, t);


% Define the radius of the cylinder
r = 2*mean(std(distances));

% Number of points around the circumference of the cylinder
theta = linspace(0, 2*pi, 30);

% Initialize matrices to hold the cylinder surface data
X = zeros(length(t), length(theta));
Y = zeros(length(t), length(theta));
Z = zeros(length(t), length(theta));

% Loop through each point on the curve to create the cylinder
for i = 1:length(t)
    % Define the tangent vector using backward difference (except for the first point)
    tangent = [dx(i), dy(i), dz(i)];
    
    % Normalize the tangent vector
    tangent = tangent / norm(tangent);
    
    % % Find vectors normal to the tangent
    % normal = null(tangent);  % Find two vectors normal to the tangent
    % 
    % % Create the cylinder at each point
    % X(i, :) = x_list(i) + r*(normal(1,1)*cos(theta) + normal(1,2)*sin(theta));
    % Y(i, :) = y_list(i) + r*(normal(2,1)*cos(theta) + normal(2,2)*sin(theta));
    % Z(i, :) = z_list(i) + r*(normal(3,1)*cos(theta) + normal(3,2)*sin(theta));

    if i == 1
        % Initial normal vector (can be any vector not parallel to the tangent)
        normal = [0, 0, 1];
        if abs(dot(normal, tangent)) > 0.9
            normal = [0, 1, 0];
        end
    else
        % Normal vector as the cross product of the previous binormal and the tangent
        normal = cross(binormal, tangent);
        normal = normal / norm(normal);
    end
    
    % Binormal vector as the cross product of tangent and normal
    binormal = cross(tangent, normal);

    % Create the cylinder at each point using the Frenet-Serret frame
    for j = 1:length(theta)
        X(i, j) = x_list(i) + r * (normal(1) * cos(theta(j)) + binormal(1) * sin(theta(j)));
        Y(i, j) = y_list(i) + r * (normal(2) * cos(theta(j)) + binormal(2) * sin(theta(j)));
        Z(i, j) = z_list(i) + r * (normal(3) * cos(theta(j)) + binormal(3) * sin(theta(j)));
    end
end

% Plot the 3D curve
figure;
plot3(x_list, y_list, z_list, 'k', 'LineWidth', 2);
hold on;

% Plot the cylinder around the curve, colored by t
h = surf(X, Y, Z, 'FaceAlpha', 0.7, 'EdgeColor', 'none');

% Set the color data to t for each row (t values are replicated along the columns)
h.CData = repmat(t', 1, length(theta));

% Adjust plot settings
axis equal;
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
title('3D Curve with Smooth Cylinder Colored by t');
colormap(jet);  % You can change this to other colormaps (e.g., parula, hot)
colorbar;       % Show a colorbar to indicate the mapping of t to color
grid on;
view(3);













%% functions

% exponential moving average filter
function ema = eMA(data, windowSize)
    alpha = 2 / (windowSize + 1);
    ema = zeros(size(data));
    ema(1) = data(1);

    for i = 2:length(data)
        ema(i) = alpha * data(i) + (1 - alpha) * ema(i - 1);
    end
end

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

% remove outliers
function output = remove_outliers(data, range)
if nargin == 1
    range  =  1.5;
end
Q1 = quantile(data, 0.25);
Q3 = quantile(data, 0.75);
IQR = Q3 - Q1;
lowerBound = Q1 - range * IQR;
upperBound = Q3 + range * IQR;
inliers = (data >= lowerBound) & (data <= upperBound);
output = data(inliers);
end

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

% swtest
function [H, pValue, W] = swtest(x, alpha)
%SWTEST Shapiro-Wilk parametric hypothesis test of composite normality.
%   [H, pValue, SWstatistic] = SWTEST(X, ALPHA) performs the
%   Shapiro-Wilk test to determine if the null hypothesis of
%   composite normality is a reasonable assumption regarding the
%   population distribution of a random sample X. The desired significance 
%   level, ALPHA, is an optional scalar input (default = 0.05).
%
%   The Shapiro-Wilk and Shapiro-Francia null hypothesis is: 
%   "X is normal with unspecified mean and variance."
%
%   This is an omnibus test, and is generally considered relatively
%   powerful against a variety of alternatives.
%   Shapiro-Wilk test is better than the Shapiro-Francia test for
%   Platykurtic sample. Conversely, Shapiro-Francia test is better than the
%   Shapiro-Wilk test for Leptokurtic samples.
%
%   When the series 'X' is Leptokurtic, SWTEST performs the Shapiro-Francia
%   test, else (series 'X' is Platykurtic) SWTEST performs the
%   Shapiro-Wilk test.
% 
%    [H, pValue, SWstatistic] = SWTEST(X, ALPHA)
%
% Inputs:
%   X - a vector of deviates from an unknown distribution. The observation
%     number must exceed 3 and less than 5000.
%
% Optional inputs:
%   ALPHA - The significance level for the test (default = 0.05).
%  
% Outputs:
%  SWstatistic - The test statistic (non normalized).
%
%   pValue - is the p-value, or the probability of observing the given
%     result by chance given that the null hypothesis is true. Small values
%     of pValue cast doubt on the validity of the null hypothesis.
%
%     H = 0 => Do not reject the null hypothesis at significance level ALPHA.
%     H = 1 => Reject the null hypothesis at significance level ALPHA.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Copyright (c) 17 March 2009 by Ahmed Ben Sa飀a          %
%                 Department of Finance, IHEC Sousse - Tunisia           %
%                       Email: ahmedbensaida@yahoo.com                   %
%                    $ Revision 3.0 $ Date: 18 Juin 2014 $               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%
% References:
%
% - Royston P. "Remark AS R94", Applied Statistics (1995), Vol. 44,
%   No. 4, pp. 547-551.
%   AS R94 -- calculates Shapiro-Wilk normality test and P-value
%   for sample sizes 3 <= n <= 5000. Handles censored or uncensored data.
%   Corrects AS 181, which was found to be inaccurate for n > 50.
%   Subroutine can be found at: http://lib.stat.cmu.edu/apstat/R94
%
% - Royston P. "A pocket-calculator algorithm for the Shapiro-Francia test
%   for non-normality: An application to medicine", Statistics in Medecine
%   (1993a), Vol. 12, pp. 181-184.
%
% - Royston P. "A Toolkit for Testing Non-Normality in Complete and
%   Censored Samples", Journal of the Royal Statistical Society Series D
%   (1993b), Vol. 42, No. 1, pp. 37-43.
%
% - Royston P. "Approximating the Shapiro-Wilk W-test for non-normality",
%   Statistics and Computing (1992), Vol. 2, pp. 117-119.
%
% - Royston P. "An Extension of Shapiro and Wilk's W Test for Normality
%   to Large Samples", Journal of the Royal Statistical Society Series C
%   (1982a), Vol. 31, No. 2, pp. 115-124.
%

%
% Ensure the sample data is a VECTOR.
%

if numel(x) == length(x)
    x  =  x(:);               % Ensure a column vector.
else
    error(' Input sample ''X'' must be a vector.');
end

%
% Remove missing observations indicated by NaN's and check sample size.
%

x  =  x(~isnan(x));

if length(x) < 3
   error(' Sample vector ''X'' must have at least 3 valid observations.');
end

if length(x) > 5000
    warning('Shapiro-Wilk test might be inaccurate due to large sample size ( > 5000).');
end

%
% Ensure the significance level, ALPHA, is a 
% scalar, and set default if necessary.
%

if (nargin >= 2) && ~isempty(alpha)
   if ~isscalar(alpha)
      error(' Significance level ''Alpha'' must be a scalar.');
   end
   if (alpha <= 0 || alpha >= 1)
      error(' Significance level ''Alpha'' must be between 0 and 1.'); 
   end
else
   alpha  =  0.05;
end

% First, calculate the a's for weights as a function of the m's
% See Royston (1992, p. 117) and Royston (1993b, p. 38) for details
% in the approximation.

x       =   sort(x); % Sort the vector X in ascending order.
n       =   length(x);
mtilde  =   norminv(((1:n)' - 3/8) / (n + 1/4));
weights =   zeros(n,1); % Preallocate the weights.

if kurtosis(x) > 3
    
    % The Shapiro-Francia test is better for leptokurtic samples.
    
    weights =   1/sqrt(mtilde'*mtilde) * mtilde;

    %
    % The Shapiro-Francia statistic W' is calculated to avoid excessive
    % rounding errors for W' close to 1 (a potential problem in very
    % large samples).
    %

    W   =   (weights' * x)^2 / ((x - mean(x))' * (x - mean(x)));

    % Royston (1993a, p. 183):
    nu      =   log(n);
    u1      =   log(nu) - nu;
    u2      =   log(nu) + 2/nu;
    mu      =   -1.2725 + (1.0521 * u1);
    sigma   =   1.0308 - (0.26758 * u2);

    newSFstatistic  =   log(1 - W);

    %
    % Compute the normalized Shapiro-Francia statistic and its p-value.
    %

    NormalSFstatistic =   (newSFstatistic - mu) / sigma;
    
    % Computes the p-value, Royston (1993a, p. 183).
    pValue   =   1 - normcdf(NormalSFstatistic, 0, 1);
    
else
    
    % The Shapiro-Wilk test is better for platykurtic samples.

    c    =   1/sqrt(mtilde'*mtilde) * mtilde;
    u    =   1/sqrt(n);

    % Royston (1992, p. 117) and Royston (1993b, p. 38):
    PolyCoef_1   =   [-2.706056 , 4.434685 , -2.071190 , -0.147981 , 0.221157 , c(n)];
    PolyCoef_2   =   [-3.582633 , 5.682633 , -1.752461 , -0.293762 , 0.042981 , c(n-1)];

    % Royston (1992, p. 118) and Royston (1993b, p. 40, Table 1)
    PolyCoef_3   =   [-0.0006714 , 0.0250540 , -0.39978 , 0.54400];
    PolyCoef_4   =   [-0.0020322 , 0.0627670 , -0.77857 , 1.38220];
    PolyCoef_5   =   [0.00389150 , -0.083751 , -0.31082 , -1.5861];
    PolyCoef_6   =   [0.00303020 , -0.082676 , -0.48030];

    PolyCoef_7   =   [0.459 , -2.273];

    weights(n)   =   polyval(PolyCoef_1 , u);
    weights(1)   =   -weights(n);
    
    if n > 5
        weights(n-1) =   polyval(PolyCoef_2 , u);
        weights(2)   =   -weights(n-1);
    
        count  =   3;
        phi    =   (mtilde'*mtilde - 2 * mtilde(n)^2 - 2 * mtilde(n-1)^2) / ...
                (1 - 2 * weights(n)^2 - 2 * weights(n-1)^2);
    else
        count  =   2;
        phi    =   (mtilde'*mtilde - 2 * mtilde(n)^2) / ...
                (1 - 2 * weights(n)^2);
    end
        
    % Special attention when n = 3 (this is a special case).
    if n == 3
        % Royston (1992, p. 117)
        weights(1)  =   1/sqrt(2);
        weights(n)  =   -weights(1);
        phi = 1;
    end

    %
    % The vector 'WEIGHTS' obtained next corresponds to the same coefficients
    % listed by Shapiro-Wilk in their original test for small samples.
    %

    weights(count : n-count+1)  =  mtilde(count : n-count+1) / sqrt(phi);

    %
    % The Shapiro-Wilk statistic W is calculated to avoid excessive rounding
    % errors for W close to 1 (a potential problem in very large samples).
    %

    W   =   (weights' * x) ^2 / ((x - mean(x))' * (x - mean(x)));

    %
    % Calculate the normalized W and its significance level (exact for
    % n = 3). Royston (1992, p. 118) and Royston (1993b, p. 40, Table 1).
    %

    newn    =   log(n);

    if (n >= 4) && (n <= 11)
    
        mu      =   polyval(PolyCoef_3 , n);
        sigma   =   exp(polyval(PolyCoef_4 , n));    
        gam     =   polyval(PolyCoef_7 , n);
    
        newSWstatistic  =   -log(gam-log(1-W));
    
    elseif n > 11
    
        mu      =   polyval(PolyCoef_5 , newn);
        sigma   =   exp(polyval(PolyCoef_6 , newn));
    
        newSWstatistic  =   log(1 - W);
    
    elseif n == 3
        mu      =   0;
        sigma   =   1;
        newSWstatistic  =   0;
    end

    %
    % Compute the normalized Shapiro-Wilk statistic and its p-value.
    %

    NormalSWstatistic   =   (newSWstatistic - mu) / sigma;
    
    % NormalSWstatistic is referred to the upper tail of N(0,1),
    % Royston (1992, p. 119).
    pValue       =   1 - normcdf(NormalSWstatistic, 0, 1);
    
    % Special attention when n = 3 (this is a special case).
    if n == 3
        pValue  =   6/pi * (asin(sqrt(W)) - asin(sqrt(3/4)));
        % Royston (1982a, p. 121)
    end
    
end

%
% To maintain consistency with existing Statistics Toolbox hypothesis
% tests, returning 'H = 0' implies that we 'Do not reject the null 
% hypothesis at the significance level of alpha' and 'H = 1' implies 
% that we 'Reject the null hypothesis at significance level of alpha.'
%

H  = (alpha >= pValue);
end