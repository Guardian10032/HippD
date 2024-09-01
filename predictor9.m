% evaluation for 30, all <150 neuron:

% train ratio: 0.01, 0.05, 0.1, 0.2, 0.3, 1
% neruon: 20, 50, 100, 150


clear
load hc_30_processed.mat
tic;
train_ratio = 1;
neuron = 100;
result(length(hc_30)) = struct('input_f', [], 'output_d', [], 'output_e', [], 'corr', [], 'R2', [], 'RMSE', []);
parfor trial=1:length(hc_30)

    intensity_raw = hc_30(trial).traces;
    speed = hc_30(trial).speed;


    % eMA of intensity
    intensity_all=zeros(size(intensity_raw));
    for i =1:size(intensity_raw,2)
        intensity_all(:,i) = eMA(intensity_raw(:,i),40);
    end

    % Random input and testing set
    [intensity_t, speed_t, input, output_d]=training_set(intensity_all, speed, train_ratio, neuron);
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
    for n=1:mean_group_valid
        for i=1:3
            distances_std(n,i) = std(intensity_g{1,n}(:,i));
        end
    end

    % predict -> with eMA filter and extended curve
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
    output_e = output_raw * (speed_r(2) - speed_r(1)) + speed_r(1);

    result(trial).input_f=input_f;
    result(trial).output_d=output_d;
    result(trial).output_e=output_e;
    result(trial).corr=corr(output_e,output_d);
    result(trial).R2=fitlm(output_e,output_d).Rsquared.Ordinary;
    result(trial).RMSE=sqrt(mean((output_e - output_d) .^ 2));
    result(trial).distances_std=distances_std;

    fprintf('trial %d is done\n',trial)
end
toc;