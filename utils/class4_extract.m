

for i = 1:20
filename = sprintf('class4_run%d.csv',i);
csvwrite(filename,EEG(i).data)
end