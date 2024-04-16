classdef LanzerNet
    properties
        sigmoid=@(z) 1./(1+exp(-z));
        dsigmoid=@(z) (1./(1+exp(-z))).*(1-(1./(1+exp(-z))));
        relu=@(z) z.*sign(z);
        drelu=@(z) sign(z);
        activate;
        dactivate;
        mu=0;
        sigma=0.1;
    end

    methods
        function net = LanzerNet()
            %init class
        end

        function this=set_activate_function(this,method)
            %set activate function
            switch method
                case 'sigmoid'
                    this.activate=this.sigmoid;
                    this.dactivate=this.dsigmoid;
                case 'relu'
                    this.activate=this.relu;
                    this.dactivate=this.drelu;
            end
        end

        function [weight1,weight2,bias1,bias2]=weight_init(this,X,Y,num_hiddens)
            %三层神经网络初始化权重和偏置
            num_features=size(X,2);
            num_labels=size(Y,2);
            weight1=normrnd(this.mu,this.sigma,[num_hiddens,num_features]);
            weight2=normrnd(this.mu,this.sigma,[num_labels,num_hiddens]);
            bias1=normrnd(this.mu,this.sigma,[num_hiddens,1]);
            bias2=normrnd(this.mu,this.sigma,[num_labels,1]);
        end

        function [weight,bias]=net_init(this,X,Y,num_hiddens,method)
            %任意层神经网络初始化
            this=this.set_activate_function(method);
            num_hidden_layers=size(num_hiddens,1);
            num_features=size(X,2);
            num_labels=size(Y,2);
            weight=cell([num_hidden_layers+1,1]);
            bias=cell([num_hidden_layers+1,1]);
            weight{1}=normrnd(this.mu,this.sigma,[num_hiddens(1),num_features]);
            bias{1}=normrnd(this.mu,this.sigma,[num_hiddens(1),1]);
            for i=2:num_hidden_layers
                weight{i}=normrnd(this.mu,this.sigma,[num_hiddens(i),num_hiddens(i-1)]);
                bias{i}=normrnd(this.mu,this.sigma,[num_hiddens(i),1]);
            end
            weight{num_hidden_layers+1}=normrnd(this.mu,this.sigma,[num_labels,num_hiddens(num_hidden_layers)]);
            bias{num_hidden_layers+1}=normrnd(this.mu,this.sigma,[num_labels,1]);
        end

        function output=forward(this,input,weight,bias)
            %任意层神经网络前向传播(推理)
            num_examples=size(input,1);
            num_layers=size(weight,1);
            num_labels=size(weight{num_layers},1);
            u=cell([num_layers,1]);
            x=cell([num_layers,1]);
            output=zeros([num_examples,num_labels]);
            for i=1:num_examples
                %forward propagation
                u{1}=weight{1}*input(i,:)'+bias{1};
                x{1}=this.relu(u{1});
                for k=2:num_layers
                    u{k}=weight{k}*x{k-1}+bias{k};
                    x{k}=this.relu(u{k});
                end
                output(i,:)=x{k}';
            end
        end

        function [weight,bias,loss]=backward(this,X,Y,weight,bias,learning_rate)
            %任意层神经网络反向传播
            num_examples=size(X,1);
            num_layers=size(weight,1);
            u=cell([num_layers,1]);
            x=cell([num_layers,1]);
            z=cell([num_layers,1]);
            for i=1:num_examples
                %forward propagation
                u{1}=weight{1}*X(i,:)'+bias{1};
                x{1}=this.relu(u{1});
                for k=2:num_layers
                    u{k}=weight{k}*x{k-1}+bias{k};
                    x{k}=this.relu(u{k});
                end
                %backward propagation
                z{num_layers}=(x{num_layers}-Y(i,:)').*this.drelu(u{k});
                for k=num_layers-1:-1:1
                    z{k}=(weight{k+1}'*z{k+1}).*this.drelu(u{k});
                end
                %update weight
                weight{1}=weight{1}-learning_rate*(z{1}*X(i,:));
                bias{1}=bias{1}-learning_rate*z{1};
                for k=2:num_layers
                    weight{k}=weight{k}-learning_rate*(z{k}*x{k-1}');
                    bias{k}=bias{k}-learning_rate*z{k};
                end
            end
            output=this.forward(X,weight,bias);
            loss=0.5*sum((Y-output).^2,"all");
        end

        function weight=gradient_descent(~,features,labels,weight,learning_rate)
            %梯度下降
            num_examples=size(features,1);
            m=size(features,2);    
            for j=1:m
                delta_weight=0;
                for i=1:num_examples
                    prediction=features(i,:)*weight;
                    error=labels(i)-prediction;
                    delta_weight=delta_weight+learning_rate*error*features(i,j)/num_examples;
                end
                weight(j)=weight(j)+delta_weight;
            end
        end

        function weight=stochastic_gradient_descent(~,features,labels,weight,learning_rate)
            %随机梯度下降
            num_examples=size(features,1);
            m=size(features,2); 
            for i=1:num_examples
                prediction=features(i,:)*weight;
                for j=1:m
                    delta_weight=learning_rate*(labels(i)-prediction)*features(i,j);
                    weight(j)=weight(j)+delta_weight;
                end
            end        
        end

        function [weight1,weight2,bias1,bias2,loss]=backward_propagation(this,X,Y,weight1,weight2,bias1,bias2,learning_rate)
            %三层神经网络单样本反向传播
            num_examples=size(X,1);
            for i=1:num_examples
                x=X(i,:)';
                z=Y(i,:)';
                u1=weight1*x+bias1;
                y=this.sigmoid(u1);
                u2=weight2*y+bias2;
                delta_weight2=((this.sigmoid(weight2*y+bias2)-z).*(this.dsigmoid(weight2*y+bias2)))*y';
                delta_bias2=(this.sigmoid(weight2*y+bias2)-z).*(this.dsigmoid(weight2*y+bias2));
                delta_weight1=(weight2'*(this.sigmoid(u2)-z).*this.dsigmoid(u2)).*(this.dsigmoid(u1)*x');
                delta_bias1=(weight2'*(this.sigmoid(u2)-z).*this.dsigmoid(u2)).*this.dsigmoid(u1);
                weight1=weight1-learning_rate*delta_weight1;
                bias1=bias1-learning_rate*delta_bias1;
                weight2=weight2-learning_rate*delta_weight2;
                bias2=bias2-learning_rate*delta_bias2;
            end
            output=this.forward_propagation(X,weight1,weight2,bias1,bias2);
            loss=0.5*sum((Y-output).^2,"all");
        end

        function output=forward_propagation(this,X,weight1,weight2,bias1,bias2)
            %三层神经网络前向传播
            num_examples=size(X,1);
            num_labels=size(weight2,1);
            output=zeros([num_examples,num_labels]);
            for i=1:num_examples
                output(i,:)=(this.sigmoid(weight2*this.sigmoid(weight1*X(i,:)'+bias1)+bias2))';
            end
        end

        function weight=train(this,features,labels,weight,learning_rate,epoch,method)
            %训练
            switch(method)
                case "GD"
                    for i=1:epoch
                        weight=this.gradient_descent(features,labels,weight,learning_rate);
                        learning_rate=0.9*learning_rate;
                    end
                case "SGD"
                    for i=1:epoch
                        weight=this.stochastic_gradient_descent(features,labels,weight,learning_rate);
                    end
            end
        end

        function outputs=prediction(~,X,weight)
            outputs=zeros(size(X,1),1);
            for i=1:size(X,1)
                outputs(i)=X(i,:)*weight;
            end
        end
    end
end