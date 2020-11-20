import torch
import torch.nn as nn
import torch.nn.init as init
import pdb
class Vgg_face_dag(nn.Module):

	def __init__(self, return_layer):
		super(Vgg_face_dag, self).__init__()
		self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
		 'std': [1, 1, 1],
		 'imageSize': [224, 224, 3]}
		self.return_layer = return_layer
		self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
		self.relu1_1 = nn.ReLU(inplace=True)
		self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
		self.relu1_2 = nn.ReLU(inplace=True)
		self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
		self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
		self.relu2_1 = nn.ReLU(inplace=True)
		self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
		self.relu2_2 = nn.ReLU(inplace=True)
		self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
		self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
		self.relu3_1 = nn.ReLU(inplace=True)
		self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
		self.relu3_2 = nn.ReLU(inplace=True)
		self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
		self.relu3_3 = nn.ReLU(inplace=True)
		self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
		self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
		self.relu4_1 = nn.ReLU(inplace=True)
		self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
		self.relu4_2 = nn.ReLU(inplace=True)
		self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
		self.relu4_3 = nn.ReLU(inplace=True)
		self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
		self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
		self.relu5_1 = nn.ReLU(inplace=True)
		self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
		self.relu5_2 = nn.ReLU(inplace=True)
		self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
		self.relu5_3 = nn.ReLU(inplace=True)
		self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
		self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
		self.relu6 = nn.ReLU(inplace=True)
		self.dropout6 = nn.Dropout(p=0.5)
		self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
		self.relu7 = nn.ReLU(inplace=True)
		self.dropout7 = nn.Dropout(p=0.5)
		self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

	def forward(self, x0):
		x1 = self.conv1_1(x0)
		x2 = self.relu1_1(x1)
		x3 = self.conv1_2(x2)
		x4 = self.relu1_2(x3)
		x5 = self.pool1(x4)
		x6 = self.conv2_1(x5)
		x7 = self.relu2_1(x6)
		x8 = self.conv2_2(x7)
		x9 = self.relu2_2(x8)
		x10 = self.pool2(x9)
		x11 = self.conv3_1(x10)
		x12 = self.relu3_1(x11)
		x13 = self.conv3_2(x12)
		x14 = self.relu3_2(x13)
		x15 = self.conv3_3(x14)
		x16 = self.relu3_3(x15)
		x17 = self.pool3(x16)
		x18 = self.conv4_1(x17)
		x19 = self.relu4_1(x18)
		x20 = self.conv4_2(x19)
		x21 = self.relu4_2(x20)
		x22 = self.conv4_3(x21)
		x23 = self.relu4_3(x22)
		x24 = self.pool4(x23)
		x25 = self.conv5_1(x24)
		x26 = self.relu5_1(x25)
		x27 = self.conv5_2(x26)
		x28 = self.relu5_2(x27)
		x29 = self.conv5_3(x28)
		x30 = self.relu5_3(x29)
		x31_preflatten = self.pool5(x30)
		x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
		x32 = self.fc6(x31)
		x33 = self.relu6(x32)
		x34 = self.dropout6(x33)
		x35 = self.fc7(x34)
		x36 = self.relu7(x35)
		x37 = self.dropout7(x36)
		x38 = self.fc8(x37)
		if self.return_layer == 'conv':
			return x31
		elif self.return_layer == 'fc6':
			return x33
		elif self.return_layer == 'fc7':
			return x36
		else:
			return x38

def vgg_face_dag(weights_path=None, return_layer='fc8',**kwargs):
	"""
	load imported model instance

	Args:
	weights_path (str): If set, loads model weights from the given path
	"""
	model = Vgg_face_dag(return_layer)

	if weights_path:
		state_dict = torch.load(weights_path, map_location=torch.device('cuda'))
		#pdb.set_trace()
		try:
			model.load_state_dict(state_dict)
		except:
			from collections import OrderedDict
			odict = OrderedDict()
			for k in state_dict:
				new_k = k.replace("module.", "")
				odict[new_k] = state_dict[k]
			#print("h1",odict.keys())
			model.load_state_dict(odict)
	return model

def spoof_model(input_dim=4096, weights_path=None):

	model = nn.Sequential(
		nn.Linear(in_features=input_dim, out_features=8192, bias=True),
		nn.ReLU(inplace=True),
		nn.Linear(in_features=8192, out_features=1000, bias=True),
		nn.ReLU(inplace=True),
		nn.Linear(in_features=1000, out_features=500, bias=True),
		nn.ReLU(inplace=True),
		nn.Linear(in_features=500, out_features=2, bias=True)
		)

	if weights_path:
		state_dict = torch.load(weights_path)
		#print(state_dict.keys())
		try:
			model.load_state_dict(state_dict)
		except:
			from collections import OrderedDict
			odict = OrderedDict()
			for k in state_dict:
				new_k = k.replace("module.", "")
				odict[new_k] = state_dict[k]
			#print("h2", odict.keys())
			#print("arrrrr");exit()
			model.load_state_dict(odict)
	return model

class decoder_nn(nn.Module):

	def __init__(self):
		super(decoder_nn, self).__init__()
		self.up1  = nn.Sequential(
			nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
			# nn.BatchNorm2d(ngf*8),
			nn.InstanceNorm2d(256),
			nn.ReLU(True),
			# nn.Upsample(scale_factor=7)
			)
		self.up2  = nn.Sequential(
			nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
			# nn.BatchNorm2d(ngf*8),
			nn.InstanceNorm2d(64),
			nn.ReLU(True),
			# nn.Upsample(scale_factor=2)
			)

		self.up3  = nn.Sequential(
			nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),
			# nn.BatchNorm2d(ngf*8),
			nn.InstanceNorm2d(16),
			nn.ReLU(True),
			# nn.Upsample(scale_factor=2)
			)

		self.up4  = nn.Sequential(
			nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
			# nn.Upsample(scale_factor=1),
			# nn.Tanh()
			# nn.ReLU(True)
			)
		self.tanh = nn.Tanh()

	def forward(self,x):
		out = x
		# print(out.shape)
		out = self.up1(out)
		# print(out.shape)
		out = self.up2(out)
		# print(out.shape)
		out = self.up3(out)
		# print(out.shape)
		out = self.up4(out)
		out = self.tanh(out)
		# print(out.shape)
		# exit()
		return out


def weight_init(m):
	# print("bitch")
	if isinstance(m, nn.Linear):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
