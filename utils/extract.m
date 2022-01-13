

prompt = 'What is label(1~4)? ';
x = input(prompt);
60
for i = 1:60
filename = sprintf('class%d_run%d.csv',x,i);
csvwrite(filename,EEG(i).data)
end