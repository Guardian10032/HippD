clear
%%
speeds={};
for i=1:length(event_traces)
    position=event_traces(i).position(:,1);
    speed = abs(diff(position));
    speed = speed - min(speed);
    speed = [0; speed];
    speed = speed / max(speed);
    speeds{1,i}=speed;
end
%%
clear
load 30\event_traces.mat
hc_30 = struct();
%%
min_length=34000;
index = 1:length(event_traces);
% remove because less than 34000 samples 
% after cutting leading or tail errors
index([6,8,28,33,35,36,37,39,47,52,72,73,79,80, 81,83,84,85,100,102,103])=[];
% change the size to reduce the influence of length
% and remove errors by visual inspection
ranges = repmat([1, min_length], length(index), 1);
ranges(index==12,:)=[15,15+min_length];
ranges(index==13,:)=[1,1+min_length];
ranges(index==14,:)=[55,55+min_length];
ranges(index==19,:)=[1233,1233+min_length];
ranges(index==20,:)=[826,826+min_length];
ranges(index==53,:)=[1159,1159+min_length];
ranges(index==60,:)=[560,560+min_length];
ranges(index==68,:)=[890,890+min_length];
ranges(index==69,:)=[88,88+min_length];

%%
for i=1:length(index)
    % infor
    hc_30(i).original_index=index(i);
    hc_30(i).session_ID=event_traces(index(i)).session_ID;

    % speed
    position=event_traces(index(i)).position(:,1);
    speed = abs(diff(position));
    speed = speed - min(speed);
    speed = [0; speed];
    speed = speed(ranges(i,1):ranges(i,2));
    speed = speed / max(speed);
    hc_30(i).speed=speed;

    % traces
    traces = event_traces(index(i)).traces(ranges(i,1):ranges(i,2),:);
    hc_30(i).traces=traces;
    % % after EMA filter, too large to save
    % traces_filtered=zeros(size(traces));
    % for n =1:size(traces,2)
    %     traces_filtered(:,i) = eMA(traces(:,i),40);
    % end
    % hc_30(i).traces_filtered=traces_filtered;
end
%%
save("hc_30_processed","hc_30")











%%
trial=3;

intensity_m = event_traces(trial).traces;
% eMA
intensity_all=zeros(size(intensity_m));
for i =1:size(intensity_m,2)
    intensity_all(:,i) = eMA(intensity_m(:,i),100);
end

position=event_traces(trial).position(:,1);
speed = abs(diff(position));
speed = speed - min(speed);
speed = [0; speed];
speed = speed / max(speed);
