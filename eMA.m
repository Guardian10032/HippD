% exponential moving average filter
function ema = eMA(data, windowSize)
alpha = 2 / (windowSize + 1);
ema = zeros(size(data));
ema(1) = data(1);

for i = 2:length(data)
    ema(i) = alpha * data(i) + (1 - alpha) * ema(i - 1);
end
end