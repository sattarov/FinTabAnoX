from torch import nn
import torch

class Encoder(nn.Module):
    def __init__(self, hidden_size, bottleneck_fnc):
        super(Encoder, self).__init__()
        # init encoder architecture
        self.linear_layers = self.init_layers(hidden_size)
        self.bottleneck_fnc = bottleneck_fnc
        self.lrelu_layer = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.relu_layer = nn.ReLU(inplace=True)
        self.tanh_layer = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def init_layers(self, layer_dimensions):
        
        layers = []
        for i in range(len(layer_dimensions)-1):
            linear_layer = self.linear_layer(layer_dimensions[i], layer_dimensions[i + 1])
            layers.append(linear_layer)
            
            self.add_module('linear_' + str(i), linear_layer)
        return layers
        
    def linear_layer(self, input_size, hidden_size):
        linear = nn.Linear(input_size, hidden_size, bias=True)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.constant_(linear.bias, 0.0)
        return linear 
    
    def forward(self, x):
        # Define the forward pass
        for i in range(len(self.linear_layers)):
            if i < len(self.linear_layers)-1:
                x = self.relu_layer(self.linear_layers[i](x))
            else:
                if self.bottleneck_fnc == 'lrelu':
                    x = self.lrelu_layer(self.linear_layers[i](x))
                elif self.bottleneck_fnc == 'relu':
                    x = self.relu_layer(self.linear_layers[i](x))
                elif self.bottleneck_fnc == 'tanh':
                    x = self.tanh_layer(self.linear_layers[i](x))
                elif self.bottleneck_fnc == 'sigmoid':
                    x = self.sigmoid(self.linear_layers[i](x))
                else:
                    print('WRONG bottleneck function name !!!')

        return x

class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        # init encoder architecture
        self.linear_layers = self.init_layers(hidden_size)
        self.lrelu_layer = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.relu_layer = nn.ReLU(inplace=True)
        self.tanh_layer = nn.Tanh()
        self.sigmoid_layer = nn.Sigmoid()

    def init_layers(self, layer_dimensions):
        
        layers = []
        for i in range(len(layer_dimensions)-1):
            linear_layer = self.linear_layer(layer_dimensions[i], layer_dimensions[i + 1])
            layers.append(linear_layer)
            
            self.add_module('linear_' + str(i), linear_layer)
        return layers
        
    def linear_layer(self, input_size, hidden_size):
        linear = nn.Linear(input_size, hidden_size, bias=True)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.constant_(linear.bias, 0.0)
        return linear 
    
    def forward(self, x):
        # Define the forward pass
        for i in range(len(self.linear_layers)):
            if i <  len(self.linear_layers)-1:
                x = self.relu_layer(self.linear_layers[i](x))
            else:
                # x = self.sigmoid_layer(self.linear_layers[i](x))
                # x = self.relu_layer(self.linear_layers[i](x))
                x = self.linear_layers[i](x)

        return x

class AutoEncoder(nn.Module):
    def __init__(self, encoder_layers, decoder_layers, bottleneck_fnc='lrelu'):
        super(AutoEncoder, self).__init__()
        
        self.encoder = Encoder(encoder_layers, bottleneck_fnc=bottleneck_fnc)
        self.decoder = Decoder(decoder_layers)
    
    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z

