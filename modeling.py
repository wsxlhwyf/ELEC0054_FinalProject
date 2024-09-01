import torch
import torch.nn as nn
import torch.nn.functional as F

class ShareBottom(nn.Module):
    """
    feature extraction
    """
    def __init__(self):
        super(ShareBottom, self).__init__()

        # using CNN
        self.conv1 = nn.Conv1d(3, 64, 1, stride=1)
        self.conv2 = nn.Conv1d(64, 128, 1, stride=1)
        self.conv3 = nn.Conv1d(128, 512, 1, stride=1)
        self.conv4 = nn.Conv1d(512, 1024, 1, stride=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)

        # using LSTM
        # self.bilstm1 = nn.LSTM(input_size=3, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        # self.bilstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        # self.bilstm3 = nn.LSTM(input_size=128, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True)
        # self.bn1_lstm = nn.BatchNorm1d(64)
        # self.bn2_lstm = nn.BatchNorm1d(128)
        # self.bn3_lstm = nn.BatchNorm1d(1024)


        # using transformer
        # self.transformer_cnn = nn.TransformerEncoderLayer(d_model=1024, nhead=1, dim_feedforward=2048, batch_first=True)
        # self.transformer_lstm = nn.TransformerEncoderLayer(d_model=1024, nhead=1, dim_feedforward=2048, batch_first=True)
        # self.bn1_tr = nn.BatchNorm1d(1024)
        # self.bn2_tr = nn.BatchNorm1d(1024)

    def forward(self, x):
        # cnn infer
        cnn_out = F.relu(self.bn1(self.conv1(torch.transpose(x, 1, 2))))
        cnn_out = F.relu(self.bn2(self.conv2(cnn_out)))
        cnn_out = F.relu(self.bn3(self.conv3(cnn_out)))
        cnn_out = self.bn4(self.conv4(cnn_out))
        # cnn_out = torch.transpose(cnn_out, 1, 2)
        # cnn_out = self.transformer_cnn(cnn_out)
        # cnn_out = self.bn1_tr(torch.transpose(cnn_out, 1, 2))
        cnn_out_max = torch.max(cnn_out, 2, keepdim=False)[0]
        cnn_out_mean = torch.mean(cnn_out, 2, keepdim=False)
        cnn_out_min = torch.min(cnn_out, 2, keepdim=False)[0]
        cnn_out_std = torch.std(cnn_out, 2, keepdim=False)


        # # bilstm infer
        # lstm_out, (h_n, c_n) = self.bilstm1(x)
        # lstm_out = torch.transpose(F.relu(self.bn1_lstm(torch.transpose(lstm_out, 1 ,2))), 1, 2)
        # lstm_out, (h_n, c_n) = self.bilstm2(lstm_out)
        # lstm_out = torch.transpose(F.relu(self.bn2_lstm(torch.transpose(lstm_out, 1, 2))), 1, 2)
        # lstm_out, (h_n, c_n) = self.bilstm3(lstm_out)
        # lstm_out = torch.transpose(F.relu(self.bn3_lstm(torch.transpose(lstm_out, 1, 2))), 1, 2)
        # # lstm_out = self.transformer_lstm(lstm_out)
        # # lstm_out = torch.transpose(self.bn2_tr(torch.transpose(lstm_out, 1, 2)), 1, 2)
        # lstm_out = torch.max(lstm_out, 2, keepdim=True)[0]

        return cnn_out_max, cnn_out_mean, cnn_out_min, cnn_out_std

class ShareBottomCls(nn.Module):
    """
    Declare the rest network architecture

    """
    def __init__(self, num_classes=40):
        super(ShareBottomCls, self).__init__()
        self.feat1 = ShareBottom()
        self.feat2 = ShareBottom()
        self.feat3 = ShareBottom()
        self.feat4 = ShareBottom()
        self.feat5 = ShareBottom()
        self.feat6 = ShareBottom()

        self.fc1_max = nn.Linear(1024, 512)
        self.fc2_max = nn.Linear(512, 128)
        self.fc3_max = nn.Linear(128, num_classes)
        self.dropout_max = nn.Dropout(p=0.1)
        self.bn1_max = nn.BatchNorm1d(512)
        self.bn2_max = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

        self.fc1_mean = nn.Linear(1024, 512)
        self.fc2_mean = nn.Linear(512, 128)
        self.fc3_mean = nn.Linear(128, num_classes)
        self.dropout_mean = nn.Dropout(p=0.1)
        self.bn1_mean = nn.BatchNorm1d(512)
        self.bn2_mean = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

        self.fc1_min = nn.Linear(1024, 512)
        self.fc2_min = nn.Linear(512, 128)
        self.fc3_min = nn.Linear(128, num_classes)
        self.dropout_min = nn.Dropout(p=0.1)
        self.bn1_min = nn.BatchNorm1d(512)
        self.bn2_min = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

        self.fc1_std = nn.Linear(1024, 512)
        self.fc2_std = nn.Linear(512, 128)
        self.fc3_std = nn.Linear(128, num_classes)
        self.dropout_std = nn.Dropout(p=0.1)
        self.bn1_std = nn.BatchNorm1d(512)
        self.bn2_std = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

        self.transformer = nn.TransformerEncoderLayer(d_model=40, nhead=4, dim_feedforward=64, batch_first=True)

    def forward(self, x):
        cnn_max1, cnn_mean1, cnn_min1, cnn_std1 = self.feat1(x)
        # cnn_out2 = self.feat2(x)
        # cnn_out3 = self.feat3(x)
        # cnn_out4 = self.feat4(x)
        # cnn_out5 = self.feat5(x)
        # cnn_out6 = self.feat6(x)

        # cnn_out = torch.concat([cnn_max1, cnn_mean1, cnn_min1, cnn_std1], dim=1)
        # cnn_out = cnn_out.view(-1, 4, 1024)
        # trans_out = self.transformer(cnn_out)
        # trans_out = torch.max(trans_out, dim=1, keepdim=False)[0]


        out_max = F.relu(self.bn1_max(self.fc1_max(cnn_max1)))
        out_max = F.relu(self.bn2_max(self.dropout_max(self.fc2_max(out_max))))
        out_max = self.fc3_max(out_max)
        out_max = F.log_softmax(out_max, dim=-1)

        out_min = F.relu(self.bn1_min(self.fc1_min(cnn_mean1)))
        out_min = F.relu(self.bn2_min(self.dropout_min(self.fc2_min(out_min))))
        out_min = self.fc3_min(out_min)
        out_min = F.log_softmax(out_min, dim=-1)

        out_mean = F.relu(self.bn1_mean(self.fc1_mean(cnn_mean1)))
        out_mean = F.relu(self.bn2_mean(self.dropout_mean(self.fc2_mean(out_mean))))
        out_mean = self.fc3_mean(out_mean)
        out_mean = F.log_softmax(out_mean, dim=-1)

        out_std = F.relu(self.bn1_std(self.fc1_std(cnn_std1)))
        out_std = F.relu(self.bn2_std(self.dropout_std(self.fc2_std(out_std))))
        out_std = self.fc3_std(out_std)
        out_std = F.log_softmax(out_std, dim=-1)

        concat = torch.concat([out_max, out_min, out_mean, out_std], dim=1).view(-1, 4, 40)
        concat = F.log_softmax(self.transformer(concat), dim=-1)



        return torch.mean(concat, dim=1, keepdim=False)