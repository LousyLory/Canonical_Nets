clear all; close all; clc;
load('arrdata.mat');
count = 0;
for i = 1: size(accuracy,1)
    maxacc = max(accuracy(i,1), accuracy(i,2));
    minacc = min(accuracy(i,1), accuracy(i,2));
    if accuracy(i,5)>=minacc && accuracy(i,5)<=maxacc
        count = count+1;
    end
end

figure(1),
ax1 = axes('Position',[0 0 1 1],'Visible','off');
ax2 = axes('Position',[.3 .1 .6 .8]);

plot(ax2, accuracy); title('validation accuracy of different models using canonicalization (w/o nomralization) vs number of runs');
legend('model1 accuracy', 'model2 accuracy', '1+2 accuracy', '1+2(greedy) accuracy', '1+2(hun) accuracy');
xlabel('number of runs'), ylabel('validation accuracy');
descr={'Number of cases in which ';
    'accuracy of Hungarian lies';
    'within bounds: '; num2str(count)};
axes(ax1), text(0.05, 0.8, descr);

%%
load('arrdata_new_canning.mat');
count = 0;
for i = 1: size(accuracy,1)
    maxacc = max(accuracy(i,1), accuracy(i,2));
    minacc = min(accuracy(i,1), accuracy(i,2));
    if accuracy(i,5)>=minacc && accuracy(i,5)<=maxacc
        count = count+1;
    end
end

figure(2),
ax1 = axes('Position',[0 0 1 1],'Visible','off');
ax2 = axes('Position',[.3 .1 .6 .8]);

plot(ax2, accuracy); title('validation accuracy of different models using canonicalization (with nomralization) vs number of runs');
legend('model1 accuracy', 'model2 accuracy', '1+2 accuracy', '1+2(greedy) accuracy', '1+2(hun) accuracy');
xlabel('number of runs'), ylabel('validation accuracy');
descr={'Number of cases in which ';
    'accuracy of Hungarian lies';
    'within bounds: '; num2str(count)};
axes(ax1), text(0.05, 0.8, descr);
