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