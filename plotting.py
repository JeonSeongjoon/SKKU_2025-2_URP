import os
import pdb
import matplotlib.pyplot as plt

from data import load_jsonl

file_name = 'log_kakaocorp-kanana-1.5-8b-instruct-2505_epochs:5_lr:0.0001.jsonl'
file_path = os.path.join('./result/log', file_name)

path_info = os.path.dirname(file_path)
log_info = file_path.removeprefix(path_info)

logData = load_jsonl(file_path)

# plotting
train_step = []; train_loss = []
test_step = []; test_loss = []

for ind, line in enumerate(logData):
    if 'step' in line.keys():
        train_step.append(line['step'])
        train_loss.append(line['train_loss'])
    else:
        test_step.append(logData[ind-1]['step'])
        test_loss.append(line['test_loss'])

X_tr = train_step; Y_tr = train_loss
X_ts = test_step; Y_ts = test_loss

#pdb.set_trace()

plt.plot(X_tr, Y_tr, color='blue', label='Train_loss')
plt.scatter(X_ts, Y_ts, color='green', label='Test_loss')
plt.title(f'{log_info}')
plt.xlabel('step'); plt.ylabel('loss')
plt.savefig('train_history_fig.png')
plt.show()


