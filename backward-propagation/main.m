clear
clc
num_hiddens=[3;5;3];
data=Data();
net=LanzerNet();
num_examples=1000;
epochs=200;
loss=zeros(epochs,1);
learning_rate=0.06;
[X,Y]=data.synthetic_nolinear_data(num_examples);
[X,Y,data]=data.normalization(X,Y);
[W,b]=net.net_init(X,Y,num_hiddens,'relu');
for epoch=1:epochs
    [W,b,loss(epoch)]=net.backward(X,Y,W,b,learning_rate);
end
O=net.forward(X,W,b);
clf
hold on
data.data_scatter(X(:,1),Y);
data.data_scatter(X(:,1),O);
hold off